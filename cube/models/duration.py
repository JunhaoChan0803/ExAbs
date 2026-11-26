from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

DURATION_SUPPORT = (5, 10, 15, 20, 25, 35, 50)


@dataclass(frozen=True)
class CubeDurationPosteriorOutput:
    probabilities: torch.Tensor
    map_duration: torch.Tensor
    conservative_duration: torch.Tensor
    normalized_entropy: torch.Tensor
    nearest_distance: torch.Tensor
    fallback: torch.Tensor


class DurationPosterior(torch.nn.Module):
    """Frozen ridge-Gaussian posterior q(duration | macro latent)."""

    def __init__(
        self,
        *,
        linear_coefficients: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        prediction_sigma: float,
        temperature: float,
        ood_distance_threshold: float,
        entropy_threshold: float,
        fallback_duration: int = 5,
    ) -> None:
        super().__init__()
        coefficients = torch.as_tensor(linear_coefficients, dtype=torch.float32)
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        if coefficients.shape != (33,):
            raise ValueError("Cube duration ridge coefficients must have shape [33]")
        if mean.shape != (32,) or std.shape != (32,):
            raise ValueError("duration normalizer must have 32 latent dimensions")
        if not all(torch.isfinite(value).all() for value in (coefficients, mean, std)):
            raise ValueError("duration posterior tensors must be finite")
        if (std <= 0).any():
            raise ValueError("duration posterior standard deviations must be positive")
        if prediction_sigma <= 0 or temperature <= 0 or ood_distance_threshold <= 0:
            raise ValueError("posterior scales and OOD threshold must be positive")
        if not 0 <= entropy_threshold <= 1:
            raise ValueError("entropy threshold must lie in [0,1]")
        if fallback_duration not in DURATION_SUPPORT:
            raise ValueError("fallback duration is outside Cube support")
        self.register_buffer("linear_coefficients", coefficients.contiguous())
        self.register_buffer("mean", mean.contiguous())
        self.register_buffer("std", std.contiguous())
        self.prediction_sigma = float(prediction_sigma)
        self.temperature = float(temperature)
        self.ood_distance_threshold = float(ood_distance_threshold)
        self.entropy_threshold = float(entropy_threshold)
        self.fallback_duration = int(fallback_duration)

    @classmethod
    def fit_from_bank(
        cls,
        *,
        macro_latents: torch.Tensor,
        durations: torch.Tensor,
        temperature: float = 1.0,
        ood_quantile: float = 0.995,
        entropy_threshold: float = 0.92,
    ) -> "DurationPosterior":
        actions = (
            torch.as_tensor(macro_latents, dtype=torch.float32).reshape(-1, 32).cpu()
        )
        labels_env = torch.as_tensor(durations, dtype=torch.long).reshape(-1).cpu()
        if actions.shape[0] != labels_env.shape[0]:
            raise ValueError("macro latents and duration labels do not align")
        if not 0.5 < ood_quantile < 1:
            raise ValueError("OOD quantile must lie in (0.5,1)")
        mean = actions.mean(dim=0)
        std = actions.std(dim=0, unbiased=False).clamp_min(1.0e-5)
        normalized = (actions - mean) / std
        design = torch.cat(
            (
                normalized.double(),
                torch.ones((len(normalized), 1), dtype=torch.float64),
            ),
            dim=1,
        )
        ridge = torch.eye(33, dtype=torch.float64) * 1.0e-4
        ridge[-1, -1] = 0.0
        coefficients = torch.linalg.solve(
            design.T @ design + ridge, design.T @ labels_env.double()
        )
        prediction = design @ coefficients
        residual = labels_env.double() - prediction
        prediction_sigma = max(float(residual.square().mean().sqrt()), 0.5)
        radial = normalized.square().mean(dim=1).sqrt()
        threshold = max(float(torch.quantile(radial, ood_quantile)), 1.0e-5)
        return cls(
            linear_coefficients=coefficients.float(),
            mean=mean,
            std=std,
            prediction_sigma=prediction_sigma,
            temperature=temperature,
            ood_distance_threshold=threshold,
            entropy_threshold=entropy_threshold,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "duration_support": list(DURATION_SUPPORT),
            "linear_coefficients": self.linear_coefficients.detach().cpu(),
            "mean": self.mean.detach().cpu(),
            "std": self.std.detach().cpu(),
            "prediction_sigma": self.prediction_sigma,
            "temperature": self.temperature,
            "ood_distance_threshold": self.ood_distance_threshold,
            "entropy_threshold": self.entropy_threshold,
            "fallback_duration": self.fallback_duration,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DurationPosterior":
        if (
            tuple(int(item) for item in payload.get("duration_support", ()))
            != DURATION_SUPPORT
        ):
            raise ValueError("Cube duration support differs")
        return cls(
            linear_coefficients=payload["linear_coefficients"],
            mean=payload["mean"],
            std=payload["std"],
            prediction_sigma=float(payload["prediction_sigma"]),
            temperature=float(payload["temperature"]),
            ood_distance_threshold=float(payload["ood_distance_threshold"]),
            entropy_threshold=float(payload["entropy_threshold"]),
            fallback_duration=int(payload["fallback_duration"]),
        )

    @torch.inference_mode()
    def forward(self, macro_latents: torch.Tensor) -> CubeDurationPosteriorOutput:
        latents = torch.as_tensor(
            macro_latents, dtype=torch.float32, device=self.mean.device
        )
        if latents.ndim < 2 or latents.shape[-1] != 32:
            raise ValueError("macro latents must end in 32 dimensions")
        leading = tuple(latents.shape[:-1])
        flat = (latents.reshape(-1, 32) - self.mean) / self.std
        design = torch.cat(
            (flat, torch.ones((len(flat), 1), device=flat.device)), dim=1
        )
        predicted_duration = design @ self.linear_coefficients
        support_float = torch.tensor(
            DURATION_SUPPORT, device=flat.device, dtype=flat.dtype
        )
        logits = (
            -0.5
            * (
                (predicted_duration[:, None] - support_float[None])
                / self.prediction_sigma
            ).square()
        )
        probability = torch.softmax(logits / self.temperature, dim=-1)
        nearest = flat.square().mean(dim=1).sqrt()
        safe = probability.clamp_min(1.0e-8)
        entropy = -(safe * safe.log()).sum(-1) / torch.log(
            torch.tensor(float(len(DURATION_SUPPORT)), device=safe.device)
        )
        fallback = nearest > self.ood_distance_threshold
        fallback_index = DURATION_SUPPORT.index(self.fallback_duration)
        fallback_probability = torch.nn.functional.one_hot(
            torch.full_like(fallback, fallback_index, dtype=torch.long),
            num_classes=len(DURATION_SUPPORT),
        ).float()
        probability = torch.where(fallback[:, None], fallback_probability, probability)
        support = torch.tensor(
            DURATION_SUPPORT, device=probability.device, dtype=torch.long
        )
        map_duration = support[probability.argmax(dim=-1)]
        lower_index = (
            (probability.cumsum(-1) < 0.25).sum(-1).clamp_max(len(DURATION_SUPPORT) - 1)
        )
        conservative = support[lower_index]
        return CubeDurationPosteriorOutput(
            probabilities=probability.reshape(*leading, -1),
            map_duration=map_duration.reshape(leading),
            conservative_duration=conservative.reshape(leading),
            normalized_entropy=entropy.reshape(leading),
            nearest_distance=nearest.reshape(leading),
            fallback=fallback.reshape(leading),
        )


__all__ = [
    "CubeDurationPosteriorOutput",
    "DURATION_SUPPORT",
    "DurationPosterior",
]
