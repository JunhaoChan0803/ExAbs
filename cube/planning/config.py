from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Method = Literal["world_model", "am", "petal", "exabs"]


@dataclass(frozen=True)
class CostConfig:
    """Weights of the AM and PETAL terms in the Cube planning costs."""

    method: Method
    low_action_block_env_steps: int = 5
    am_risk_weight: float = 0.5
    am_value_weight: float = 0.0
    am_lcb_standard_deviations: float = 1.0
    petal_alpha: float = 0.25
    petal_residual_clip: float = 3.0

    def __post_init__(self) -> None:
        if self.method not in {"world_model", "am", "petal", "exabs"}:
            raise ValueError(f"Unsupported method {self.method!r}")


def _current_latent(value: torch.Tensor, *, samples: int, name: str) -> torch.Tensor:
    """Recover one current latent per environment from the Hi-LeWM tensor layouts."""
    if value.ndim == 2:
        return value
    if value.ndim == 3:
        # A second axis of size S denotes samples; otherwise it denotes history.
        return value[:, 0] if value.shape[1] == samples else value[:, -1]
    if value.ndim == 4:
        if value.shape[1] not in (1, samples):
            raise ValueError(f"{name} has an unsupported sample axis: {value.shape}")
        return value[:, 0, -1]
    raise ValueError(f"{name} must have 2, 3, or 4 dimensions, got {value.shape}")
