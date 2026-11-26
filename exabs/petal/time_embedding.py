from __future__ import annotations

import torch
from torch import nn


class ScalarFourierEmbedding(nn.Module):
    """Fixed-frequency sine/cosine embedding of a bounded scalar such as duration."""

    def __init__(self, dimension: int, maximum_value: int) -> None:
        super().__init__()
        if dimension % 2:
            raise ValueError("The embedding dimension must be even")
        frequencies = 2.0 ** torch.arange(dimension // 2, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.maximum_value = float(maximum_value)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = (
            value.to(torch.float32).clamp(min=0.0, max=self.maximum_value).unsqueeze(-1)
            / self.maximum_value
        )
        angle = torch.pi * normalized * self.frequencies
        return torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)
