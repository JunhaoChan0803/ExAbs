"""Transformer encoder from variable-length action chunks to macro-actions."""

import torch
from torch import nn
from torch.nn import functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # PyTorch < 2.3
    SDPBackend = None
    sdpa_kernel = None


class TransformerCLSMacroActionPosterior(nn.Module):
    """Map padded action chunks [B,T,A] with a validity mask to Gaussian latents.

    A CLS token summarizes the valid, left-aligned actions of each chunk.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        z_dim: int,
        min_std: float,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_seq_len: int,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.z_dim = int(z_dim)
        self.min_std = float(min_std)
        self.max_seq_len = int(max_seq_len)

        self.action_projection = nn.Linear(self.action_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.max_seq_len + 1, d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * self.z_dim),
        )
        self.mu_ln = nn.LayerNorm(self.z_dim)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, actions: torch.Tensor, action_mask: torch.Tensor):
        batch_size, steps, _ = actions.shape
        mask = torch.as_tensor(action_mask, device=actions.device, dtype=torch.bool)
        tokens = self.action_projection(actions)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.position_embedding[:, : steps + 1]

        cls_is_valid = torch.ones(
            batch_size, 1, dtype=torch.bool, device=actions.device
        )
        padding = ~torch.cat([cls_is_valid, mask], dim=1)
        if actions.is_cuda and sdpa_kernel is not None:
            # The math kernel keeps these short sequences deterministic on GPU.
            with sdpa_kernel(SDPBackend.MATH):
                encoded = self.transformer(tokens, src_key_padding_mask=padding)
        else:
            encoded = self.transformer(tokens, src_key_padding_mask=padding)
        mu, std = self.output_head(encoded[:, 0]).chunk(2, dim=-1)
        mu = self.mu_ln(mu)
        std = F.softplus(std) + self.min_std
        return mu, std
