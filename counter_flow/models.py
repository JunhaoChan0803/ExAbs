"""Visual world models, action-grounded support and learned effect comparison."""

import torch
from torch import nn

from exabs.achievability import AchievabilityConfig, AchievabilityModel
from exabs.macro_encoder import TransformerCLSMacroActionPosterior
from exabs.petal import CompactEffectEncoderConfig, EffectMetricConfig, PETALCore
from exabs.selector import ActionGroundedSelectorConfig, ActionGroundedSupportSelector

from .networks.distributions import PosteriorContinuous
from .networks.encoder import MeNet6
from .networks.encoder_config import BackboneConfig
from .networks.layers import build_conv
from .networks.predictor import ConvPredictor, ConvPredictorConfig
from .networks.predictor_config import PredictorConfig

CONV_LAYERS_CONFIG = {
    "a": [
        (-1, 32, 3, 1, 1),
        ("max_pool", 2, 2, 0),
        (32, 32, 3, 1, 1),
        ("max_pool", 2, 2, 0),
        (32, 32, 3, 1, 1),
        ("fc", -1, 2),
    ]
}


def image_tensor(images, device):
    x = torch.as_tensor(images, device=device)
    if x.dtype != torch.uint8:
        raise TypeError("Encoder expects original RGB uint8")
    return x.movedim(-1, -3).float().div(255.0)


class LowModel(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = BackboneConfig(arch="menet6", backbone_subclass="d4rl_a", channels=3)
        self.encoder = MeNet6(cfg, (3, 98, 98), 0, 0)
        self.encoder.input_dim = (3, 98, 98)
        self.encoder.output_dim = tuple(
            self.encoder(torch.randn(1, 3, 98, 98)).encodings.shape[1:]
        )
        self.encoder.output_obs_dim = self.encoder.output_dim
        self.encoder.output_proprio_dim = 0
        self.encoder.output_loc_dim = 0
        self.shape = tuple(self.encoder.output_dim)
        pcfg = PredictorConfig(
            predictor_arch="conv2",
            predictor_subclass="a",
            action_encoder_arch="id",
            residual=True,
            prefused_repr=False,
            z_dim=0,
        )
        self.predictor = ConvPredictor(
            pcfg, self.shape, action_dim=2, pred_obs_dim=self.shape
        )
        self.position_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(self.shape[0] * 64, 128),
            nn.SiLU(),
            nn.Linear(128, 2),
            nn.Sigmoid(),
        )
        self.idm = build_conv(
            CONV_LAYERS_CONFIG["a"], (self.shape[0] * 2, *self.shape[1:])
        )

    def position(self, z):
        return self.position_head(z.float())

    def encode(self, x):
        return self.encoder(x).encodings

    def step(self, z, a):
        return self.predictor(z, a, z, None)

    def rollout(self, z, actions, all_steps=False):
        zs = []
        for t in range(actions.shape[1]):
            z = self.step(z, actions[:, t])
            zs.append(z)
        return torch.stack(zs, 1) if all_steps else z


def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def autoregressive_rollout(step, initial, macros):
    current = initial
    states = []
    for j in range(macros.shape[1]):
        current = step(current, macros[:, j])
        states.append(current)
    return torch.stack(states, 1)


class FixedHWM(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = tuple(shape)
        self.macro = PosteriorContinuous(
            input_dim=20,
            arch="32-32",
            z_dim=8,
            min_std=0.05,
            posterior_input_type="actions",
        )
        cfg = PredictorConfig(
            predictor_arch="conv2",
            predictor_subclass="a",
            action_encoder_arch="id",
            residual=True,
            prefused_repr=False,
            z_dim=0,
        )
        self.predictor = ConvPredictor(
            cfg, self.shape, action_dim=8, pred_obs_dim=self.shape
        )
        self.predictor.layers = build_conv(
            ConvPredictorConfig["l2_d4rl_e_p"],
            input_dim=(self.shape[0] + 8, *self.shape[1:]),
            output_dim=self.shape,
            group_factor=8,
        )

    def encode_actions(self, actions):
        if actions.shape[-2:] != (10, 2):
            raise ValueError("Each macro must encode exactly ten 2D actions")
        leading = actions.shape[:-2]
        mean, _ = self.macro(actions.reshape(-1, 20))
        return mean.reshape(*leading, 8)

    def step(self, state, macro):
        return self.predictor(state, macro, state, None)

    def forward(self, initial, actions):
        if actions.shape[1:] != (4, 10, 2):
            raise ValueError("Four macro transitions are required")
        return autoregressive_rollout(self.step, initial, self.encode_actions(actions))


class SelectorTraining(nn.Module):
    def __init__(self):
        super().__init__()
        self.selector = make_selector(16)
        self.decoder = nn.Sequential(
            nn.Linear(49, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 28),
        )

    def forward(self, source, target, mask, coverage):
        support = (
            self.selector.straight_through_support
            if self.training
            else self.selector.hard_support
        )(source, target, coverage)
        w = support / support.sum((-2, -1), keepdim=True).clamp_min(1)
        features = torch.cat(
            [
                (source * w).sum((-2, -1)),
                (target * w).sum((-2, -1)),
                ((target - source) * w).sum((-2, -1)),
                mask.sum(-1, keepdim=True).float() / 14,
            ],
            -1,
        )
        return self.decoder(features).reshape(-1, 14, 2)


def petal_model():
    return PETALCore(
        encoder_config=CompactEffectEncoderConfig(
            feature_channels=16,
            effect_dim=96,
            hidden_channels=128,
            time_embedding_dim=32,
        ),
        metric_config=EffectMetricConfig(
            effect_dim=96, hidden_dim=256, executor_anchor_weight=0.9
        ),
    )


def achievability_model():
    return AchievabilityModel(
        AchievabilityConfig(feature_channels=16, maximum_budget=30, maximum_gap=14)
    )


class SegmentedHigh(FixedHWM):
    def __init__(self, shape):
        super().__init__(shape)
        self.macro = TransformerCLSMacroActionPosterior(
            action_dim=2,
            z_dim=8,
            min_std=0.05,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.0,
            max_seq_len=14,
        )

    def encode_actions(self, actions, mask):
        lead = actions.shape[:-2]
        return self.macro(actions.reshape(-1, 14, 2), mask.reshape(-1, 14))[0].reshape(
            *lead, 8
        )

    def forward(self, sources, actions, mask):
        b, n = sources.shape[:2]
        macro = self.encode_actions(actions, mask)
        return self.step(sources.flatten(0, 1), macro.flatten(0, 1)).reshape(
            b, n, *self.shape
        )


def make_selector(channels):
    return ActionGroundedSupportSelector(
        ActionGroundedSelectorConfig(input_channels=channels)
    )


class AM(nn.Module):
    """Completion branch of the Achievability Model used for planning."""

    def __init__(self):
        super().__init__()
        full = achievability_model()
        self.frame = full.feasibility_frame_encoder
        self.relation = full.feasibility_relation
        self.heads = full.hazard_heads

    def forward(self, source, target):
        embeddings = self.frame(torch.cat([source, target], 0))
        a, b = embeddings.chunk(2)
        return self.encoded(a, b)

    def encoded(self, source, target):
        relation = self.relation(torch.stack([source, target], 1))
        return torch.stack([h(relation) for h in self.heads], 0).float()

    @staticmethod
    def cdf(logits):
        return 1 - torch.cumprod((1 - torch.sigmoid(logits)).clamp_min(1e-7), -1)


def calibrated(p, cal, budget=None):
    a, b = cal["alpha"], cal["beta"]
    if isinstance(a, list):
        if budget is not None:
            a, b = a[budget - 1], b[budget - 1]
        elif p.ndim < 2 or p.shape[-1] != len(a):
            raise ValueError("Explicit budget required for scalar calibrated CDF")
        else:
            a, b = p.new_tensor(a), p.new_tensor(b)
    return torch.sigmoid(a * torch.logit(p.clamp(1e-6, 1 - 1e-6)) + b)
