# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Action Chunking Transformer (DETR-VAE) model."""
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.autograd import Variable

from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

def reparametrize(mu: Tensor, logvar: Tensor) -> Tensor:
    """Sample a latent vector using reparameterization."""
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
    """Action Chunking Transformer (DETR-VAE) model."""

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

        if backbones is not None:
            if len(backbones) == 0:
                raise ValueError("backbones must be a non-empty list or None.")
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
        if backbones is not None:
            if not self.camera_names:
                raise ValueError("camera_names must be provided when using image backbones.")
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(self.proprio_dim, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(self.proprio_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32  # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)  # action to embedding
        self.encoder_joint_proj = nn.Linear(self.proprio_dim, hidden_dim)  # qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)  # latent std/var
        self.register_buffer(
            "pos_table",
            get_sinusoid_encoding_table(1 + 1 + self.chunk_size, hidden_dim),
        )  # [CLS], qpos, action_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)  # latent to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)  # proprio + latent positions

    def forward(
        self,
        qpos: Tensor,
        image: Tensor,
        env_state: Optional[Tensor],
        actions: Optional[Tensor] = None,
        is_pad: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, List[Optional[Tensor]]]:
        """Run the model and return actions, padding logits, and latent stats."""
        is_training = actions is not None  # train or val
        bs, _ = qpos.shape
        
        if is_training:
            ## Obtain latent z from encoder during trainning.
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            ## During inference, z is sampled from standard Gaussian.
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, _ in enumerate(self.camera_names):
                features, pos = self.backbones[cam_id](image[:, cam_id])
                features = features[-1]  # take the last layer feature
                pos = pos[-1]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)

            # Note: ACT paper says they use the last layer output,
            # but their code uess the first layer output.
            # The origion code is: hs = self.transformer(...)[0]
            # We change it to match the paper.
            hs = self.transformer(
                src=src,
                mask=None,
                query_embed=self.query_embed.weight,
                pos_embed=pos,
                latent_input=latent_input,
                proprio_input=proprio_input,
                additional_pos_embed=self.additional_pos_embed.weight
            )[-1]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]



def build_encoder(config: Any) -> TransformerEncoder:
    """Build a transformer encoder for the VAE branch."""
    d_model = config.hidden_dim
    dropout = config.dropout
    nhead = config.nheads
    dim_feedforward = config.dim_feedforward
    num_encoder_layers = config.enc_layers  # 4 # TODO shared with VAE decoder
    normalize_before = config.pre_norm  # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
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
