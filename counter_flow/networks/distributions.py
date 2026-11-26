import math
from typing import List, Union

import torch
from torch import nn
from torch.nn import functional as F

from .layers import build_conv


class PriorContinuous(torch.nn.Module):
    def __init__(self, input_dim: int, arch: str, z_dim: int, min_std: float):
        super().__init__()

        self.z_dim = z_dim
        self.arch = arch

        if arch != "uniform" and arch != "":
            self.input_dim = input_dim
            self.min_std = min_std
            self.prior_net = MLP(
                arch=arch, input_dim=self.input_dim, output_shape=2 * self.z_dim
            )
        else:
            self.prior_net = None

    def sample(self, stats):
        mu, var = stats
        return torch.randn_like(mu) * var + mu

    def forward(self, state_encoding, batch_dim):
        if self.arch == "uniform":
            bs = state_encoding.shape[batch_dim]
            mu = torch.zeros((bs, self.z_dim)).to(state_encoding.device)
            std = torch.ones((bs, self.z_dim)).to(state_encoding.device)
        elif self.arch == "":
            bs = state_encoding.shape[batch_dim]
            mu = torch.zeros((bs, self.z_dim)).to(state_encoding.device)
            std = torch.ones((bs, self.z_dim)).to(state_encoding.device)
        else:
            mu, std = self.prior_net(state_encoding).chunk(2, dim=-1)
            std = F.softplus(std) + self.min_std
        return mu, std


CONV_LAYERS_CONFIG = {
    "b": [
        (-1, 32, 3, 1, 1),
        ("max_pool", 2, 2, 0),
        (32, 32, 3, 1, 1),
        ("max_pool", 2, 2, 0),
        ("fc", -1, -1),
    ],
}


class PosteriorContinuous(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        arch: str,
        z_dim: int,
        min_std: float,
        posterior_input_type: str = "term_states",
        mean_layer_norm: bool = True,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.min_std = min_std
        self.posterior_input_type = posterior_input_type
        self.arch = arch
        self.mu_ln = nn.LayerNorm(self.z_dim) if mean_layer_norm else nn.Identity()

        if arch == "conv":
            self.posterior_net = build_conv(
                CONV_LAYERS_CONFIG["b"],
                input_dim=(2 * input_dim[0], *input_dim[1:]),
                output_dim=2 * self.z_dim,
            )
        else:
            if not isinstance(input_dim, int):
                input_dim = math.prod(input_dim)

            self.posterior_net = MLP(
                arch=arch,
                input_dim=input_dim,
                output_shape=2 * self.z_dim,
            )

    def sample(self, stats):
        mu, var = stats
        return torch.randn_like(mu) * var + mu

    def forward(self, input: torch.Tensor):
        """Encode paired endpoint states or an action chunk, according to input type."""
        mu, std = self.posterior_net(input).chunk(2, dim=-1)
        mu = self.mu_ln(mu)
        std = F.softplus(std) + self.min_std
        return mu, std


class IdLn(torch.nn.Module):
    def __init__(self, input_dim, z_dim: int, min_std: float):
        super().__init__()
        if not isinstance(input_dim, int):
            input_dim = math.prod(input_dim)

        if input_dim != z_dim:
            raise ValueError(
                f"IdLn requires input_dim == z_dim for identity behavior, got input_dim={input_dim}, z_dim={z_dim}"
            )

        self.input_dim = input_dim
        self.min_std = min_std

    def sample(self, stats):
        mu, var = stats
        return torch.randn_like(mu) * var + mu

    def forward(self, input: torch.Tensor):
        mu = input
        std = torch.ones_like(mu) * self.min_std
        return mu, std


class AnalyticalPosterior(torch.nn.Module):
    def __init__(self, min_std: float):
        super().__init__()
        self.min_std = min_std

    def sample(self, stats):
        mu, var = stats
        return torch.randn_like(mu) * var + mu

    def forward(self, actions: torch.Tensor):
        if actions.ndim != 3:
            raise ValueError(
                f"AnalyticalPosterior expected actions shape (bs, t, 2), got {tuple(actions.shape)}"
            )
        if actions.shape[-1] != 2:
            raise ValueError(
                f"AnalyticalPosterior expected primitive action dim 2, got {actions.shape[-1]}"
            )

        bs, t, _ = actions.shape

        action_sum = actions.sum(dim=1)
        weights = torch.arange(
            t - 1, -1, -1, device=actions.device, dtype=actions.dtype
        )
        weighted_action_sum = (actions * weights.view(1, t, 1)).sum(dim=1)

        mu = torch.cat([action_sum, weighted_action_sum], dim=-1)
        std = torch.ones_like(mu) * self.min_std
        return mu, std


def build_projector(arch: str, embedding: int):
    if arch == "id":
        return nn.Identity(), embedding
    else:
        f = [embedding] + list(map(int, arch.split("-")))
        return build_mlp(f), f[-1]


def build_norm1d(norm: str, dim: int):
    if norm == "batch_norm":
        return torch.nn.BatchNorm1d(dim)
    elif norm == "layer_norm":
        return torch.nn.LayerNorm(dim)
    else:
        raise ValueError(f"Unknown norm {norm}")


def build_activation(activation: str):
    if activation == "relu":
        return nn.ReLU(True)
    elif activation == "mish":
        return nn.Mish(True)
    else:
        raise ValueError(f"Unknown activation {activation}")


def build_mlp(
    layers_dims: Union[List[int], str],
    input_dim: int = None,
    output_shape: int = None,
    norm="batch_norm",
    activation="relu",
    pre_actnorm=False,
    post_norm=False,
    dropout: float = 0.0,
):
    if isinstance(layers_dims, str):
        layers_dims = (
            list(map(int, layers_dims.split("-"))) if layers_dims != "" else []
        )

    if input_dim is not None:
        layers_dims = [input_dim] + layers_dims

    if output_shape is not None:
        layers_dims = layers_dims + [output_shape]

    layers = []

    if pre_actnorm:
        if norm is not None:
            layers.append(build_norm1d(norm, layers_dims[0]))
        if activation is not None:
            layers.append(build_activation(activation))
        if dropout:
            layers.append(nn.Dropout(dropout))

    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        if norm is not None:
            layers.append(build_norm1d(norm, layers_dims[i + 1]))
        if activation is not None:
            layers.append(build_activation(activation))
        if dropout:
            layers.append(nn.Dropout(dropout))

    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))

    if post_norm:
        layers.append(build_norm1d(norm, layers_dims[-1]))

    return nn.Sequential(*layers)


class MLP(torch.nn.Module):
    def __init__(
        self,
        arch: str,
        input_dim: int = None,
        output_shape: int = None,
        norm=None,
        activation="relu",
    ):
        super().__init__()

        self.mlp = build_mlp(
            layers_dims=arch,
            input_dim=input_dim,
            output_shape=output_shape,
            norm=norm,
            activation=activation,
        )

    def forward(self, x):
        return self.mlp(x)


PROBER_CONV_LAYERS_CONFIG = {
    "a": [
        (-1, 16, 3, 1, 1),
        ("max_pool", 2, 2, 0),
        (16, 8, 1, 1, 0),
        ("max_pool", 2, 2, 0),
        ("fc", -1, 2),
    ],
    "b": [
        (-1, 32, 3, 1, 1),
        ("max_pool", 2, 2, 0),
        (32, 32, 3, 1, 1),
        ("max_pool", 2, 2, 0),
        (32, 32, 3, 1, 1),
        ("fc", -1, 2),
    ],
    "c": [
        (-1, 32, 3, 1, 1),
        ("max_pool", 2, 2, 0),
        (32, 32, 3, 1, 1),
        ("max_pool", 2, 2, 0),
        (32, 32, 3, 1, 1),
        (32, 32, 3, 1, 1),
        ("fc", -1, 2),
    ],
}
