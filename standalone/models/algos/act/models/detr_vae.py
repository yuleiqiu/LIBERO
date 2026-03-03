# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Action Chunking Transformer (DETR-VAE) model.

This module intentionally keeps the original parameter names for checkpoint
compatibility, while using clearer local variable names and comments so the VAE
training/inference paths are easier to follow.
"""
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.autograd import Variable

from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

def reparametrize(mu: Tensor, logvar: Tensor) -> Tensor:
    """Sample a latent vector from the posterior parameterized by ``mu``/``logvar``."""
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position: int, d_hid: int) -> Tensor:
    """Build sinusoidal positional encodings."""
    def get_position_angle_vec(position: int) -> List[float]:
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class ACTModel(nn.Module):
    """Action Chunking Transformer (DETR-VAE) model.

    The model has two execution modes:

    - Training: encode the target action chunk with a VAE encoder to obtain a
      latent sample, then decode the future action chunk conditioned on the
      current robot state and images.
    - Inference: skip the VAE encoder and use a zero latent embedding while
      decoding actions from the current observation only.

    Note: attribute/module names are kept stable to avoid checkpoint key
    changes. Readability improvements here are limited to comments and local
    variables.
    """

    def __init__(
        self,
        config: Any,
        backbones: Optional[List[nn.Module]] = None,
        transformer: Optional[nn.Module] = None,
        encoder: Optional[nn.Module] = None,
        train_backbone: bool = True,
    ) -> None:
        """Initialize the model from a config object."""
        action_dim, proprio_dim = _resolve_dims(config)
        chunk_size = int(getattr(config, "chunk_size", 0) or getattr(config, "num_queries", 0) or 0)
        camera_names = getattr(config, "camera_names", None) or []

        if chunk_size <= 0:
            raise ValueError("chunk_size must be provided and > 0")

        if backbones is None:
            backbones = [build_backbone(config, train_backbone=train_backbone)]
        if transformer is None:
            transformer = build_transformer(config)
        if encoder is None:
            encoder = build_encoder(config)

        if backbones is None or len(backbones) == 0:
            raise ValueError("ACTModel requires at least one image backbone.")
        if camera_names and len(backbones) != len(camera_names):
            raise ValueError(
                "backbones length must match camera_names length when using image inputs."
            )

        super().__init__()
        self.chunk_size = int(chunk_size)
        self.num_queries = self.chunk_size
        self.camera_names = list(camera_names)
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.proprio_dim = int(proprio_dim) if proprio_dim is not None else int(action_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(self.chunk_size, hidden_dim)
        if not self.camera_names:
            raise ValueError("camera_names must be provided when using image backbones.")
        self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
        self.backbones = nn.ModuleList(backbones)
        self.input_proj_robot_state = nn.Linear(self.proprio_dim, hidden_dim)

        # VAE encoder components: [CLS, proprio, action_chunk] -> latent posterior.
        self.latent_dim = 32  # Size of z. Kept fixed for checkpoint compatibility.
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
        self.encoder_joint_proj = nn.Linear(self.proprio_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
        self.register_buffer(
            "pos_table",
            get_sinusoid_encoding_table(1 + 1 + self.chunk_size, hidden_dim),
        )  # Positional encodings for [CLS, proprio, action_chunk].

        # Decoder conditioning components: latent embedding + proprio embedding.
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

    def forward(
        self,
        qpos: Tensor,
        image: Tensor,
        actions: Optional[Tensor] = None,
        is_pad: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, List[Optional[Tensor]]]:
        """Run the model and return action chunk, padding logits, and latent stats."""
        is_training = actions is not None
        batch_size, _ = qpos.shape
        
        if is_training:
            # Training path: encode [CLS, proprio, target_action_chunk] into a
            # latent posterior, then sample z via reparameterization.
            action_embeddings = self.encoder_action_proj(actions)
            proprio_embeddings = self.encoder_joint_proj(qpos).unsqueeze(1)
            cls_embedding = self.cls_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)

            encoder_tokens = torch.cat(
                [cls_embedding, proprio_embeddings, action_embeddings], axis=1
            )
            encoder_tokens = encoder_tokens.permute(1, 0, 2)

            # The prepended CLS/proprio tokens are always valid; only the target
            # action chunk may contain padded steps from dataset slicing.
            cls_proprio_is_pad = torch.full((batch_size, 2), False).to(qpos.device)
            encoder_is_pad = torch.cat([cls_proprio_is_pad, is_pad], axis=1)

            encoder_pos_embed = self.pos_table.clone().detach().permute(1, 0, 2)
            encoder_output = self.encoder(
                encoder_tokens,
                pos=encoder_pos_embed,
                src_key_padding_mask=encoder_is_pad,
            )
            # Only the CLS token is used to parameterize the posterior over z.
            cls_output = encoder_output[0]
            latent_stats = self.latent_proj(cls_output)
            mu = latent_stats[:, : self.latent_dim]
            logvar = latent_stats[:, self.latent_dim :]
            latent_sample = reparametrize(mu, logvar)
            latent_embedding = self.latent_out_proj(latent_sample)
        else:
            # Inference path: no target action chunk is available, so we skip
            # the VAE encoder and decode with a zero latent embedding.
            mu = logvar = None
            latent_sample = torch.zeros(
                [batch_size, self.latent_dim], dtype=torch.float32
            ).to(qpos.device)
            latent_embedding = self.latent_out_proj(latent_sample)

        # Encode each camera independently, then concatenate feature maps
        # across cameras along the width dimension, matching original ACT.
        camera_features = []
        camera_positions = []
        for cam_id, _ in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            feature_map = features[-1]
            position_map = pos[-1]
            camera_features.append(self.input_proj(feature_map))
            camera_positions.append(position_map)

        proprio_embedding = self.input_proj_robot_state(qpos)
        image_features = torch.cat(camera_features, axis=3)
        image_positions = torch.cat(camera_positions, axis=3)

        # The ACT paper describes using the last decoder layer output. The
        # original reference code indexes the first layer instead. We keep
        # the paper-aligned behavior here.
        decoder_hidden_states = self.transformer(
            src=image_features,
            mask=None,
            query_embed=self.query_embed.weight,
            pos_embed=image_positions,
            latent_input=latent_embedding,
            proprio_input=proprio_embedding,
            additional_pos_embed=self.additional_pos_embed.weight,
        )[-1]

        predicted_actions = self.action_head(decoder_hidden_states)
        predicted_is_pad = self.is_pad_head(decoder_hidden_states)
        return predicted_actions, predicted_is_pad, [mu, logvar]



def build_encoder(config: Any) -> TransformerEncoder:
    """Build a transformer encoder for the VAE branch."""
    d_model = config.hidden_dim
    dropout = config.dropout
    nhead = config.nheads
    dim_feedforward = config.dim_feedforward
    num_encoder_layers = config.enc_layers
    normalize_before = config.pre_norm
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(
        d_model, nhead, dim_feedforward, dropout, activation, normalize_before
    )
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def _resolve_dims(config: Any) -> Tuple[int, int]:
    action_dim = getattr(config, "action_dim", None)
    if action_dim is None:
        action_dim = getattr(config, "state_dim", 14)
    proprio_dim = getattr(config, "proprio_dim", None)
    if proprio_dim is None:
        proprio_dim = getattr(config, "qpos_dim", None)
    if proprio_dim is None:
        proprio_dim = action_dim
    return int(action_dim), int(proprio_dim)
