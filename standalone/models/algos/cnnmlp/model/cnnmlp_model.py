import torch
from torch import nn


class CNNMLPModel(nn.Module):
    def __init__(self, config, backbones=None):
        """CNN-MLP baseline model."""
        action_dim, proprio_dim = _resolve_dims(config)
        camera_names = getattr(config, "camera_names", None) or []

        if backbones is None or len(backbones) == 0:
            raise ValueError("backbones must be provided for CNNMLPModel.")
        if not camera_names:
            raise ValueError("camera_names must be provided for CNNMLPModel.")
        if len(backbones) != len(camera_names):
            raise ValueError("CNNMLPModel expects one backbone per camera.")

        super().__init__()
        self.camera_names = list(camera_names)
        self.proprio_dim = int(proprio_dim) if proprio_dim is not None else int(action_dim)
        self.action_dim = int(action_dim)
        self.action_head = nn.Linear(1000, action_dim)  # TODO add more

        self.backbones = nn.ModuleList(backbones)
        backbone_down_projs = []
        for backbone in backbones:
            down_proj = nn.Sequential(
                nn.Conv2d(backbone.num_channels, 128, kernel_size=5, padding=2),
                nn.Conv2d(128, 64, kernel_size=5, padding=2),
                nn.Conv2d(64, 32, kernel_size=5, padding=2),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            backbone_down_projs.append(down_proj)
        self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

        per_cam_dim = 32 * 4 * 4
        mlp_in_dim = per_cam_dim * len(backbones) + self.proprio_dim
        self.mlp = _mlp(
            input_dim=mlp_in_dim,
            hidden_dim=1024,
            output_dim=self.action_dim,
            hidden_depth=2,
        )

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, proprio_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        bs, _ = qpos.shape
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0]  # take the last layer feature
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1)
        features = torch.cat([flattened_features, qpos], axis=1)
        a_hat = self.mlp(features)
        return a_hat


def _resolve_dims(config):
    action_dim = getattr(config, "action_dim", None)
    if action_dim is None:
        action_dim = getattr(config, "state_dim", 14)
    qpos_dim = getattr(config, "qpos_dim", None)
    if qpos_dim is None:
        qpos_dim = action_dim
    return int(action_dim), int(qpos_dim)


def _mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk
