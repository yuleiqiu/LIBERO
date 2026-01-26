# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR-VAE model for action prediction in robotics. 
This module contains a Variational Autoencoder (VAE) based on DETR architecture 
for predicting action sequences from multi-modal observations.
"""

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from quest.algos.baseline_modules.act_utils.transformer import TransformerEncoder, TransformerEncoderLayer


def reparametrize(mu, logvar):
    """
    Reparameterization trick for VAE: sample from N(mu, exp(logvar/2)) using N(0,1).
    This allows backpropagation through the sampling operation.
    
    Args:
        mu: mean of the latent distribution
        logvar: log variance of the latent distribution
        
    Returns:
        Sampled latent vector z
    """
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    """
    Generate sinusoidal positional encoding table.
    Uses alternating sine and cosine functions at different frequencies.
    
    Args:
        n_position: maximum sequence length
        d_hid: hidden dimension
        
    Returns:
        torch.FloatTensor: positional encoding table of shape (1, n_position, d_hid)
    """
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """ 
    DETR-VAE model for action prediction in robotics. Modified from DETR.
    Uses a transformer encoder to encode action sequences into a latent space (VAE),
    and a transformer decoder to predict future actions from observations and latent codes.
    """
    def __init__(self, 
                 transformer, 
                 encoder, 
                 state_dim, 
                 proprio_dim, 
                 num_queries, 
                 shape_meta,
                 encoder_input=('lowdim',)
                 ):
        """
        Initializes the DETR-VAE model.

        Parameters:
            transformer: torch module of the transformer decoder architecture for action prediction
            encoder: torch module of the transformer encoder for latent encoding
            state_dim: dimensionality of the robot action space
            proprio_dim: dimensionality of proprioceptive observations (joint positions, etc.)
            num_queries: number of action queries (maximum sequence length for action prediction)
            shape_meta: metadata containing observation space information (lowdim, rgb, etc.)
            encoder_input: tuple specifying which observation types to use in encoder. Default: ('lowdim',)
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.encoder = encoder
        self.encoder_input = encoder_input
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # VAE encoder components
        self.latent_dim = 32 # dimension of latent variable z
        self.cls_embed = nn.Embedding(1, hidden_dim) # CLS token embedding for encoder
        self.encoder_action_proj = nn.Linear(state_dim, hidden_dim) # project actions to hidden dimension
        self.encoder_joint_proj = nn.Linear(proprio_dim, hidden_dim)  # project proprioception to hidden dimension, not used
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project encoder output to latent mean and logvar
        
        # Count encoder input modalities
        n_inputs = 0
        obs_meta = shape_meta['observation']
        if 'lowdim' in encoder_input:
            if obs_meta['lowdim'] is not None:
                n_inputs += len(obs_meta['lowdim'])
        if 'perception' in encoder_input:
            if obs_meta['rgb'] is not None:
                n_inputs += len(obs_meta['rgb'])

        self.n_encoder_inputs = n_inputs
        # Positional encoding table: [CLS token, encoder inputs, action sequence]
        self.register_buffer(
            'pos_table', get_sinusoid_encoding_table(1 + num_queries + n_inputs, hidden_dim)
        )

        # VAE decoder components
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to hidden dimension
        # Count decoder input modalities (task embedding + latent + observations)
        n_decoder_inputs = 0
        if obs_meta['lowdim'] is not None:
            n_decoder_inputs += len(obs_meta['lowdim'])
        if obs_meta['rgb'] is not None:
            n_decoder_inputs += len(obs_meta['rgb'])
        self.additional_pos_embed = nn.Embedding(2 + n_decoder_inputs, hidden_dim) # learned positional embeddings for decoder inputs

    def forward(self, lowdim_encodings, perception_encodings, task_emb, actions=None, is_pad=None):
        """
        Forward pass of DETR-VAE model.
        
        Args:
            lowdim_encodings: batch, seq, hidden_dim - encoded low-dimensional observations (proprioception, etc.)
            perception_encodings: batch, seq, hidden_dim - encoded visual observations (camera images)
            task_emb: batch, hidden_dim - task embedding (instruction or task identifier)
            actions: batch, seq, action_dim - ground truth action sequence (only during training)
            is_pad: batch, seq - boolean mask indicating padded positions in action sequence
            
        Returns:
            a_hat: predicted action sequence
            is_pad_hat: predicted padding mask
            [mu, logvar]: VAE latent distribution parameters (None during inference)
        """
        is_training = actions is not None # determine training vs inference mode
        bs = lowdim_encodings.shape[0]

        # VAE Encoder: encode action sequence to latent distribution during training
        if is_training:
            # Project action sequence to hidden dimension and add CLS token
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            
            # Concatenate encoder inputs: [CLS token, observation encodings, action embeddings]
            encoder_input = [cls_embed]
            if 'lowdim' in self.encoder_input:
                encoder_input.append(lowdim_encodings)
            if 'perception' in self.encoder_input:
                encoder_input.append(perception_encodings)
            encoder_input.append(action_embed)
            encoder_input = torch.cat(encoder_input, axis=1) # (bs, seq+1+n_inputs, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1+n_inputs, bs, hidden_dim)

            # Create padding mask: don't mask CLS token and observation encodings
            cls_joint_is_pad = torch.full(
                (bs, 1 + self.n_encoder_inputs), False
            ).to(lowdim_encodings.device) # False: not padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1+n_inputs)
            
            # Apply positional encoding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1+n_inputs, 1, hidden_dim)

            # Run transformer encoder and extract latent representation from CLS token
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take CLS token output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            # During inference, use zero latent (no action sequence available)
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(lowdim_encodings.device)
            latent_input = self.latent_out_proj(latent_sample)

        # Prepare decoder inputs: [task_emb, latent, observation encodings]
        # Reshape to sequence-first format for transformer (seq, batch, hidden_dim)
        task_emb = task_emb.unsqueeze(0)
        latent_input = latent_input.unsqueeze(0)
        lowdim_encodings = lowdim_encodings.permute(1, 0, 2)
        perception_encodings = perception_encodings.permute(1, 0, 2)
        try:
            transformer_input = torch.cat([task_emb, latent_input, lowdim_encodings, perception_encodings], dim=0)
        except RuntimeError:
            raise ValueError(f"Shape mismatch in decoder inputs: task_emb: {task_emb.shape}, latent_input: {latent_input.shape}, lowdim_encodings: {lowdim_encodings.shape}, perception_encodings: {perception_encodings.shape}")

        # Prepare learnable query embeddings and positional embeddings for action prediction
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        input_pos_embed = self.additional_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        # Run transformer decoder to predict action sequence
        hs = self.transformer(transformer_input, None, query_embed, input_pos_embed)[-1] # Take final decoder layer output

        # Generate action predictions and padding predictions
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]
    
def build_encoder(
       d_model=256,
       nheads=8,
       dim_feedforward=2048,
       enc_layers=4,
       pre_norm=False,
       dropout=0.1
):
    """ 
    Build an encoder for VAE latent encoding.
    Used to build the encoder module in DETR-VAE.
    
    Args:
        d_model: model dimension
        nheads: number of attention heads
        dim_feedforward: feedforward network dimension
        enc_layers: number of encoder layers
        pre_norm: whether to use pre-normalization
        dropout: dropout rate
        
    Returns:
        TransformerEncoder: configured transformer encoder
    """
    activation = "relu"
    encoder_layer = TransformerEncoderLayer(d_model, nheads, dim_feedforward,
                                            dropout, activation, pre_norm)
    encoder_norm = nn.LayerNorm(d_model) if pre_norm else None
    encoder = TransformerEncoder(encoder_layer, enc_layers, encoder_norm)
    return encoder



class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """
        CNN + MLP policy for action prediction. Alternative to transformer-based approach.
        Not used in current experiments.

        Parameters:
            backbones: list of CNN backbone modules for processing camera images
            state_dim: dimensionality of the robot action space  
            camera_names: list of camera names corresponding to the backbones
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim) # TODO: adjust input dimension based on actual feature size
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            # Create downsampling convolutions for each camera backbone
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            # MLP input: flattened visual features + proprioceptive state
            mlp_in_dim = 768 * len(backbones) + 14
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=14, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        Forward pass for CNN+MLP policy.
        
        Args:
            qpos: batch, qpos_dim - robot joint positions/proprioceptive state
            image: batch, num_cam, channel, height, width - multi-camera images
            env_state: environment state (unused in this implementation)
            actions: batch, seq, action_dim - ground truth actions (unused, for compatibility)
            
        Returns:
            a_hat: predicted single-step action
        """
        is_training = actions is not None # determine training vs inference mode (unused here)
        bs, _ = qpos.shape
        
        # Process images from each camera
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the final layer features
            pos = pos[0] # positional encoding (not used)
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
            
        # Flatten and concatenate all visual features
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1) # concatenate features from all cameras
        
        # Combine visual features with proprioceptive state
        features = torch.cat([flattened_features, qpos], axis=1) # visual + proprioceptive features
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    """
    Build a simple fully-connected MLP network.
    Only used in CNN+MLP policy, not in current experiments.
    
    Args:
        input_dim: input feature dimension
        hidden_dim: hidden layer dimension  
        output_dim: output dimension
        hidden_depth: number of hidden layers
        
    Returns:
        nn.Sequential: MLP network
    """
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk
