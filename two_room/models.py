# Adapted from PLDM, Copyright (c) 2025 Vlad Sobal. MIT License; see licenses/PLDM.txt.
"""Visual world models and learned planning components."""

from pathlib import Path

import torch
from torch import nn

from exabs.achievability import AchievabilityConfig, AchievabilityModel
from exabs.checkpoint import load_exabs
from exabs.macro_encoder import TransformerCLSMacroActionPosterior
from exabs.petal import CompactEffectEncoderConfig, EffectMetricConfig
from exabs.selector import ActionGroundedSelectorConfig, ActionGroundedSupportSelector

from .networks.am_adapter import adapt
from .networks.convolution import Expander2D, build_conv
from .networks.scoring import AnchoredPETAL


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        config = [
            (3, 16, 5, 1, 0),
            (16, 32, 5, 2, 0),
            (32, 32, 3, 1, 0),
            (32, 32, 3, 1, 1),
            (32, 16, 1, 1, 0),
        ]
        self.layers = nn.Sequential(nn.Identity(), build_conv(config, (3,)))

    def forward(self, images):
        return self.layers(images)


class Dynamics(nn.Module):
    def __init__(self, action_dim, high=False):
        super().__init__()
        config = (
            [
                (16 + action_dim, 32, 5, 1, 2),
                (32, 32, 5, 1, 2),
                (32, 32, 5, 1, 2),
                (32, 32, 5, 1, 2),
                (32, 16, 5, 1, 2),
            ]
            if high
            else [(18, 32, 3, 1, 1), (32, 32, 3, 1, 1), (32, 16, 3, 1, 1)]
        )
        self.layers = build_conv(config, (16 + action_dim, 43, 43), group_factor=8)
        self.action_encoder = Expander2D(43, 43)

    def forward(self, z, action):
        return self.layers(torch.cat([z, self.action_encoder(action)], dim=1)) + z


class LowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape = (16, 43, 43)
        self.encoder = ImageEncoder()
        self.predictor = Dynamics(2)
        self.idm = build_conv(
            [
                (-1, 32, 3, 1, 1),
                ("max_pool", 2, 2, 0),
                (32, 32, 3, 1, 1),
                ("max_pool", 2, 2, 0),
                (32, 32, 3, 1, 1),
                ("fc", -1, 2),
            ],
            (32, 43, 43),
        )
        self.position_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(16 * 64, 128),
            nn.SiLU(),
            nn.Linear(128, 2),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def step(self, z, a):
        return self.predictor(z, a)

    def position(self, z):
        return self.position_head(z.float()).float()


class HighModel(nn.Module):
    def __init__(self, max_actions=25):
        super().__init__()
        self.macro = TransformerCLSMacroActionPosterior(
            action_dim=2,
            z_dim=8,
            min_std=0.05,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.0,
            max_seq_len=max_actions,
        )
        self.predictor = Dynamics(8, high=True)

    def encode_actions(self, actions, mask):
        return self.macro(actions, mask)[0]

    def step(self, z, macro):
        return self.predictor(z, macro)


def image_tensor(images, device):
    x = torch.as_tensor(images, device=device)
    if x.dtype != torch.uint8:
        raise TypeError("Expected RGB uint8 observations")
    return x.movedim(-1, -3).float().div(255.0)


def load_models(
    device="cpu", directory=None, *, method="exabs", runtime_directory=None
):
    """Load the components used by the selected evaluation method."""
    if method not in ("exabs", "single", "hwm"):
        raise ValueError(f"Unknown method: {method}")
    directory = Path(directory) if directory else Path(__file__).parent / "checkpoints"
    runtime_directory = (
        Path(runtime_directory)
        if runtime_directory
        else Path(__file__).parent / "assets" / "runtime"
    )
    models = {"low": LowModel()}
    paths = {"low": directory / "l1.pt"}
    if method == "hwm":
        models["high"] = HighModel(max_actions=25)
        paths["high"] = directory / "hwm.pt"
    elif method == "exabs":
        models.update(
            high=HighModel(),
            selector=ActionGroundedSupportSelector(
                ActionGroundedSelectorConfig(input_channels=16)
            ),
            petal=AnchoredPETAL(
                encoder_config=CompactEffectEncoderConfig(
                    feature_channels=16,
                    effect_dim=96,
                    hidden_channels=128,
                    time_embedding_dim=32,
                ),
                metric_config=EffectMetricConfig(
                    effect_dim=96,
                    hidden_dim=256,
                    time_embedding_dim=32,
                    executor_anchor_weight=0.9,
                ),
            ),
            am=adapt(
                AchievabilityModel(
                    AchievabilityConfig(
                        feature_channels=16, maximum_budget=25, maximum_gap=25
                    )
                )
            ),
        )
        paths.update(
            selector=runtime_directory / "selector.pt",
        )
    components = load_exabs(directory, "two_room") if method == "exabs" else {}
    keys = {"high": "l2", "am": "am", "petal": "petal"}
    for name, model in models.items():
        state = (
            components[keys[name]]
            if name in keys and method == "exabs"
            else torch.load(paths[name], map_location="cpu", weights_only=True)
        )
        model.load_state_dict(state, strict=True)
        model.to(device).eval().requires_grad_(False)
    return models
