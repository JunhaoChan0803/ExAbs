from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch


@dataclass
class PredictorConfig:
    use_vmap: bool = False
    predictor_arch: str = "conv2"
    predictor_subclass: str = "a"
    predictor_ln: bool = False
    rnn_state_dim: int = 512
    rnn_converter_arch: str = ""
    z_discrete: bool = False
    z_discrete_dim: int = 16
    z_discrete_dists: int = 16
    z_dim: int = 0
    z_min_std: float = 0.1
    z_stochastic: bool = True
    posterior_drop_p: float = 0.0
    prior_arch: str = "512"
    posterior_arch: str = "512"
    posterior_mean_layer_norm: bool = True
    posterior_input_type: str = "term_states"
    posterior_input_dim: Optional[int] = None
    # Optional duration input; disabled for existing checkpoint compatibility.
    include_duration_context: bool = False
    # Action posterior: flat MLP or masked-sequence Transformer.
    posterior_action_encoder: str = "flatten_mlp"
    posterior_transformer_d_model: int = 64
    posterior_transformer_nhead: int = 4
    posterior_transformer_num_layers: int = 2
    posterior_transformer_dim_feedforward: int = 128
    posterior_transformer_dropout: float = 0.0
    posterior_transformer_max_seq_len: int = 18
    action_encoder_arch: str = ""
    residual: bool = False
    rnn_layers: int = 1
    tie_backbone_ln: bool = False
    dropout: float = 0.0
    ensemble_size: int = 1  # when set to 1, there is no ensemble.
    prefused_repr: bool = True
    transformer_num_layers: Optional[int] = None
    transformer_activation: Optional[str] = None
    transformer_nhead: Optional[int] = None
    transformer_dim_feedforward: Optional[int] = None
    transformer_teacher_forcing_ratio: Optional[float] = None
    transformer_max_seq_len: Optional[int] = None
    use_teacher_forcing: Optional[bool] = None
    use_checkpointing: Optional[bool] = None

    def __post_init__(self):
        if self.posterior_action_encoder not in {"flatten_mlp", "transformer_cls"}:
            raise ValueError(
                "posterior_action_encoder must be 'flatten_mlp' or "
                f"'transformer_cls', got {self.posterior_action_encoder!r}."
            )
        if self.posterior_action_encoder == "transformer_cls":
            if self.posterior_input_type != "actions":
                raise ValueError(
                    "transformer_cls is only valid for an action-conditioned posterior."
                )
            if self.include_duration_context:
                raise ValueError(
                    "transformer_cls derives duration from its validity mask; "
                    "include_duration_context must remain false."
                )
            positive_fields = (
                "posterior_transformer_d_model",
                "posterior_transformer_nhead",
                "posterior_transformer_num_layers",
                "posterior_transformer_dim_feedforward",
                "posterior_transformer_max_seq_len",
            )
            for field in positive_fields:
                value = getattr(self, field)
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    raise ValueError(f"{field} must be a positive integer.")
            if self.posterior_transformer_d_model % self.posterior_transformer_nhead:
                raise ValueError(
                    "posterior_transformer_d_model must be divisible by "
                    "posterior_transformer_nhead."
                )
            if not 0.0 <= self.posterior_transformer_dropout < 1.0:
                raise ValueError("posterior_transformer_dropout must lie in [0, 1).")


class PredictorOutput(NamedTuple):
    predictions: torch.Tensor
    obs_component: Optional[torch.Tensor] = None
    proprio_component: Optional[torch.Tensor] = None
    location_component: Optional[torch.Tensor] = None
    raw_locations: Optional[torch.Tensor] = None
    ensemble_predictions: Optional[torch.Tensor] = None
    ensemble_obs_component: Optional[torch.Tensor] = None
    ensemble_proprio_component: Optional[torch.Tensor] = None
    ensemble_location_component: Optional[torch.Tensor] = None
    ensemble_raw_locations: Optional[torch.Tensor] = None
    prior_mus: Optional[torch.Tensor] = None
    prior_vars: Optional[torch.Tensor] = None
    prior_logits: Optional[torch.Tensor] = None
    priors: Optional[torch.Tensor] = None
    posterior_mus: Optional[torch.Tensor] = None
    posterior_vars: Optional[torch.Tensor] = None
    posterior_logits: Optional[torch.Tensor] = None
    posteriors: Optional[torch.Tensor] = None


class SingleStepPredictorOutput(NamedTuple):
    prediction: torch.Tensor
    obs_component: Optional[torch.Tensor] = None
    proprio_component: Optional[torch.Tensor] = None
    next_hidden_state: Optional[torch.Tensor] = None  # for RNNs
    location_component: Optional[torch.Tensor] = None
    raw_location: Optional[torch.Tensor] = None
