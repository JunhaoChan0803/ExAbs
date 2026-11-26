from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelConfig:
    """Network dimensions for the Cube AM and PETAL models.

    ``duration_mode`` and ``petal_transition_interpreter_mode`` are stored with
    the released checkpoints; the networks do not take duration as an input.
    """

    latent_dim: int = 192
    model_dim: int = 256
    attention_heads: int = 8
    relation_layers: int = 2
    candidate_set_layers: int = 2
    feedforward_dim: int = 1024
    dropout: float = 0.1
    duration_mode: str = "agnostic"
    duration_dim: int = 32
    maximum_duration_steps: float = 75.0
    hazard_bin_steps: tuple[int, ...] = tuple(range(5, 51, 5))
    ensemble_members: int = 5
    effect_dim: int = 128
    petal_hidden_dim: int = 512
    petal_metric_rank: int = 16
    metric_diagonal_floor: float = 1.0e-4
    petal_transition_interpreter_mode: str = "separate"

    def __post_init__(self) -> None:
        if self.model_dim % self.attention_heads:
            raise ValueError("model_dim must be divisible by attention_heads")
        if self.hazard_bin_steps[-1] > self.maximum_duration_steps:
            raise ValueError("The last hazard bin exceeds maximum_duration_steps")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ModelConfig":
        values = dict(raw)
        if "hazard_bin_steps" in values:
            values["hazard_bin_steps"] = tuple(values["hazard_bin_steps"])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
