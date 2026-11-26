"""Construct Cube networks and load tensor-only checkpoints."""

from pathlib import Path

import torch
from torch import nn
from transformers import ViTConfig, ViTModel

from exabs.checkpoint import load_exabs

from .components import AchievabilityModel
from .config import ModelConfig
from .duration import DurationPosterior
from .layers import MLP, ARPredictor, Embedder
from .macro_encoder import LatentActionEncoder
from .petal import PETAL
from .world_model import HiJEPA


def load_world_model(
    directory: Path, device: str = "cpu", high_filename: str | None = None
) -> HiJEPA:
    low = torch.load(directory / "l1.pt", map_location="cpu", weights_only=True)
    high = (
        load_exabs(directory, "cube")["l2"]
        if high_filename is None
        else torch.load(
            directory / high_filename, map_location="cpu", weights_only=True
        )
    )
    state = {**low["state_dict"], **high["state_dict"]}
    config = ViTConfig(**low["encoder_config"])
    config._attn_implementation = "eager"
    encoder = ViTModel(config, add_pooling_layer=False)

    def predictor(prefix):
        return ARPredictor(
            num_frames=state[prefix + ".pos_embedding"].shape[1],
            depth=6,
            heads=16,
            mlp_dim=2048,
            input_dim=192,
            hidden_dim=192,
            dim_head=64,
            dropout=0.1,
            emb_dropout=0.0,
        )

    def projection():
        return MLP(192, 2048, 192, norm_fn=nn.BatchNorm1d)

    model = HiJEPA(
        encoder=encoder,
        low_predictor=predictor("low_predictor"),
        action_encoder=Embedder(input_dim=25, smoothed_dim=10, emb_dim=192),
        high_predictor=predictor("high_predictor"),
        latent_action_encoder=LatentActionEncoder(
            input_dim=25,
            latent_dim=32,
            max_seq_len=state["latent_action_encoder.pos_embedding"].shape[1] - 1,
        ),
        macro_to_condition=nn.Linear(32, 192),
        projector=projection(),
        low_pred_proj=projection(),
        high_pred_proj=projection(),
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def load_components(directory: Path, device: str = "cpu"):
    components = load_exabs(directory, "cube")
    payload = components["am"]
    am = AchievabilityModel(ModelConfig.from_dict(payload["config"]))
    am.load_state_dict(payload["state_dict"], strict=True)
    payload = components["petal"]
    petal = PETAL(ModelConfig.from_dict(payload["config"]))
    petal.load_state_dict(payload["state_dict"], strict=True)
    payload = torch.load(
        directory.parent / "assets" / "runtime" / "duration.pt",
        map_location="cpu",
        weights_only=True,
    )
    duration = DurationPosterior.from_payload(payload)
    return tuple(
        model.to(device).eval().requires_grad_(False) for model in (am, petal, duration)
    )


def load_single_level(directory: Path, device: str = "cpu"):
    """Load the original single-level LeWM with the shared frozen L1 tensors."""
    from stable_worldmodel.wm.lewm.lewm import LeWM
    from stable_worldmodel.wm.lewm.module import Predictor

    payload = torch.load(directory / "l1.pt", map_location="cpu", weights_only=True)
    state = {}
    for name, value in payload["state_dict"].items():
        if name.startswith("low_predictor."):
            name = name.replace("low_predictor.", "predictor.", 1)
        elif name.startswith("low_pred_proj."):
            name = name.replace("low_pred_proj.", "pred_proj.", 1)
        state[name] = value
    config = ViTConfig(**payload["encoder_config"])
    config._attn_implementation = "eager"
    encoder = ViTModel(config, add_pooling_layer=False)
    predictor = Predictor(
        num_frames=state["predictor.pos_embedding"].shape[1],
        depth=6,
        heads=16,
        mlp_dim=2048,
        input_dim=192,
        hidden_dim=192,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.0,
    )
    model = LeWM(
        encoder=encoder,
        predictor=predictor,
        action_encoder=Embedder(input_dim=25, smoothed_dim=10, emb_dim=192),
        projector=MLP(192, 2048, 192, norm_fn=nn.BatchNorm1d),
        pred_proj=MLP(192, 2048, 192, norm_fn=nn.BatchNorm1d),
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)
