import torch
from torch import nn

from .layers import Expander2D, build_conv
from .predictor_base import SequencePredictor
from .predictor_config import SingleStepPredictorOutput

ConvPredictorConfig = {
    "a": [(18, 32, 3, 1, 1), (32, 32, 3, 1, 1), (32, 16, 3, 1, 1)],
    "b": [(18, 32, 5, 1, 2), (32, 32, 5, 1, 2), (32, 16, 5, 1, 2)],
    "c": [(18, 32, 7, 1, 3), (32, 32, 7, 1, 3), (32, 16, 7, 1, 3)],
    "a_proprio": [(20, 32, 3, 1, 1), (32, 32, 3, 1, 1), (32, 18, 3, 1, 1)],
    "d4rl_b_p": [(20, 32, 3, 1, 1), (32, 32, 3, 1, 1), (32, 18, 3, 1, 1)],
    "d4rl_c_p": [(20, 32, 3, 1, 1), (32, 32, 3, 1, 1), (32, 24, 3, 1, 1)],
    "d4rl_d_p": [(20, 32, 3, 1, 1), (32, 32, 3, 1, 1)],
    "d4rl_e_p": [(20, 64, 3, 1, 1), (64, 64, 3, 1, 1)],
    "l2_d4rl_b_p": [(42, 32, 3, 1, 1), (32, 32, 3, 1, 1), (32, 34, 3, 1, 1)],
    "l2_d4rl_c_p": [(42, 32, 5, 1, 2), (32, 32, 5, 1, 2), (32, 18, 5, 1, 2)],
    "l2_d4rl_e_p": [
        (42, 32, 5, 1, 2),
        (32, 32, 5, 1, 2),
        (32, 32, 5, 1, 2),
        (32, 32, 5, 1, 2),
        (32, 18, 5, 1, 2),
    ],
    "d4rl_b_p_l": [
        (20, 32, 3, 1, 1),
        (32, 32, 3, 1, 1),
        (32, 32, 3, 1, 1),
        (32, 32, 3, 1, 1),
        (32, 18, 3, 1, 1),
    ],
}


class ConvPredictor(SequencePredictor):
    def __init__(
        self,
        # parent inputs
        config,
        repr_dim,
        action_dim=2,  # action + z_dim
        pred_proprio_dim=0,
        pred_obs_dim=0,
        # child inputs
        num_groups=4,
    ):
        super(ConvPredictor, self).__init__(
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_proprio_dim=pred_proprio_dim,
            pred_obs_dim=pred_obs_dim,
        )
        self.num_groups = num_groups
        action_channels = action_dim
        if self.config.action_encoder_arch and self.config.action_encoder_arch != "id":
            action_channels = int(self.config.action_encoder_arch.split("-")[-1])
        input_dim = (
            pred_obs_dim[0]
            + (
                pred_proprio_dim
                if isinstance(pred_proprio_dim, int)
                else pred_proprio_dim[0]
            )
            + action_channels,
            *pred_obs_dim[1:],
        )

        self.layers = build_conv(
            ConvPredictorConfig[config.predictor_subclass],
            input_dim=input_dim,
            group_factor=8,
        )

        # Action encoder
        if self.config.action_encoder_arch and self.config.action_encoder_arch != "id":
            layer_dims = [int(x) for x in self.config.action_encoder_arch.split("-")]
            layers = []
            for i in range(len(layer_dims) - 1):
                layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
                layers.append(nn.ReLU())
            # remove last ReLU
            layers.pop()

            self.action_encoder = nn.Sequential(
                *layers,
                Expander2D(w=pred_obs_dim[-2], h=pred_obs_dim[-1]),
            )
        else:
            self.action_encoder = Expander2D(w=pred_obs_dim[-2], h=pred_obs_dim[-1])

    def forward_and_format(
        self, current_state, curr_action, curr_obs, curr_proprio, **kwargs
    ):
        """
        It is responsible for:
            distangle the observation and proprioception components from the state encodings
            formatting and output of the predictor.
        """

        pred_output = self.forward(
            current_state,
            curr_action=curr_action,
            curr_obs=curr_obs,
            curr_proprio=curr_proprio,
        )

        assert isinstance(pred_output, torch.Tensor), "pred_output should be a tensor"

        if self.config.ensemble_size > 1:
            obs, proprio = self._separate_obs_proprio_from_fused_repr(
                pred_output, ensemble_input=True
            )
        else:
            obs, proprio = self._separate_obs_proprio_from_fused_repr(pred_output)

        out = SingleStepPredictorOutput(
            prediction=pred_output,
            obs_component=obs,
            proprio_component=proprio,
        )

        return out

    def forward(self, current_state, curr_action, curr_obs, curr_proprio, **kwargs):
        """Predict from state [B,C,H,W], action [B,A], and optional proprioception."""
        bs, _, h, w = current_state.shape
        curr_action = self.action_encoder(curr_action)
        if curr_action.dim() == 5:
            curr_action = curr_action.flatten(1, 2)
        x = torch.cat([current_state, curr_action], dim=1)
        x = self.layers(x)
        if self.config.residual:
            x = x + current_state

        return x
