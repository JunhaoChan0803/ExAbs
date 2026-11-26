"""Spatial residual on top of a frozen AM completion branch for Two-room features."""

import torch
from torch import nn
from torch.nn import functional as F


class FramesWithMap(nn.Module):
    """Append the raw feature map [16,43,43] to the frozen frame embedding."""

    def __init__(self, base):
        super().__init__()
        self.base = base.requires_grad_(False)

    def forward(self, z):
        self.base.eval()
        with torch.no_grad():
            embedding = self.base(z)
        return torch.cat([embedding, z.flatten(1)], -1)


class SpatialRelation(nn.Module):
    """Frozen relation embedding plus a trainable source/target convolutional relation."""

    def __init__(self, base):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.register_buffer("mean", torch.zeros(1, 16, 1, 1))
        self.register_buffer("scale", torch.ones(1, 16, 1, 1))
        self.local = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1),
            nn.GroupNorm(6, 48),
            nn.SiLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.project = nn.Sequential(
            nn.Linear(2 * 64 * 6 * 6, 192), nn.SiLU(), nn.LayerNorm(192)
        )

    def forward(self, tokens):
        self.base.eval()
        with torch.no_grad():
            original = self.base(tokens[:, :, :192])
        z = tokens[:, :, 192:].reshape(-1, 2, 16, 43, 43)
        s = (z[:, 0] - self.mean) / self.scale
        t = (z[:, 1] - self.mean) / self.scale
        delta = t - s
        x = self.local(torch.cat([s, t, delta, delta.abs()], 1))
        x = torch.cat(
            [F.adaptive_avg_pool2d(x, 6), F.adaptive_max_pool2d(x, 6)], 1
        ).flatten(1)
        return torch.cat([original, self.project(x)], -1)


class ResidualHazard(nn.Module):
    """Frozen hazard head plus a zero-initialized correction from the spatial relation."""

    def __init__(self, base):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.correction = nn.Linear(192, 25)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, x):
        return self.base(x[:, :192]) + self.correction(x[:, 192:])


class FrozenReadout(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base.requires_grad_(False)

    def forward(self, x):
        return self.base(x[:, :192])


def adapt(model):
    """Wrap the completion branch of an AchievabilityModel with the spatial residual."""
    model.requires_grad_(False)
    model.feasibility_frame_encoder = FramesWithMap(model.feasibility_frame_encoder)
    model.feasibility_relation = SpatialRelation(model.feasibility_relation)
    model.hazard_heads = nn.ModuleList(
        [ResidualHazard(head) for head in model.hazard_heads]
    )
    model.residual_head = FrozenReadout(model.residual_head)
    return model
