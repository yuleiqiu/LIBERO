import torch
import torch.nn as nn

from libero.lifelong.models.modules.rgb_modules import *
from libero.lifelong.models.modules.language_modules import *
from libero.lifelong.models.bc_rnn_policy import ExtraModalities
from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.policy_head import *


###############################################################################
#
# A simple MLP BC policy (no temporal model).
#
###############################################################################


class BCMLPPolicy(BasePolicy):
    """
    Input: (o_{t-H}, ... , o_t)
    Output: a_t or distribution of a_t (per timestep, independent)
    """

    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy

        # 1. encode image (optional)
        self.image_encoders = {}
        image_embed_size = policy_cfg.image_embed_size
        mlp_input_size = 0
        for name in shape_meta["all_shapes"].keys():
            if "rgb" in name or "depth" in name:
                kwargs = policy_cfg.image_encoder.network_kwargs
                kwargs.input_shape = shape_meta["all_shapes"][name]
                kwargs.output_size = image_embed_size
                kwargs.language_dim = (
                    policy_cfg.language_encoder.network_kwargs.input_size
                )
                self.image_encoders[name] = {
                    "input_shape": shape_meta["all_shapes"][name],
                    "encoder": eval(policy_cfg.image_encoder.network)(**kwargs),
                }
                mlp_input_size += image_embed_size
        self.encoders = nn.ModuleList(
            [x["encoder"] for x in self.image_encoders.values()]
        )

        # 2. encode language
        text_embed_size = policy_cfg.text_embed_size
        policy_cfg.language_encoder.network_kwargs.output_size = text_embed_size
        self.language_encoder = eval(policy_cfg.language_encoder.network)(
            **policy_cfg.language_encoder.network_kwargs
        )
        mlp_input_size += text_embed_size

        # 3. encode extra proprio information
        self.extra_encoder = ExtraModalities(
            use_joint=cfg.data.use_joint,
            use_gripper=cfg.data.use_gripper,
            use_ee=cfg.data.use_ee,
        )
        mlp_input_size += self.extra_encoder.extra_low_level_feature_dim

        # 4. MLP trunk
        trunk_hidden = policy_cfg.mlp_hidden_size
        trunk_layers = policy_cfg.mlp_num_layers
        if trunk_layers > 0:
            sizes = [mlp_input_size] + [trunk_hidden] * trunk_layers
            layers = []
            for i in range(trunk_layers):
                layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
            self.mlp = nn.Sequential(*layers)
            head_input_size = trunk_hidden
        else:
            self.mlp = nn.Identity()
            head_input_size = mlp_input_size

        # 5. policy head
        policy_head_kwargs = policy_cfg.policy_head.network_kwargs
        policy_head_kwargs.input_size = head_input_size
        policy_head_kwargs.output_size = shape_meta["ac_dim"]
        self.policy_head = eval(policy_cfg.policy_head.network)(
            **policy_cfg.policy_head.loss_kwargs,
            **policy_cfg.policy_head.network_kwargs
        )

    def forward(self, data):
        encoded = []

        # image encoders (if any)
        for img_name in self.image_encoders.keys():
            x = data["obs"][img_name]
            B, T, C, H, W = x.shape
            e = self.image_encoders[img_name]["encoder"](
                x.reshape(B * T, C, H, W),
                langs=data["task_emb"]
                .reshape(B, 1, -1)
                .repeat(1, T, 1)
                .reshape(B * T, -1),
            ).view(B, T, -1)
            encoded.append(e)

        # proprio
        extra = self.extra_encoder(data["obs"])  # (B, T, H_extra)
        encoded.append(extra)
        B, T = extra.shape[:2]

        # language
        lang_h = self.language_encoder(data)  # (B, H_text)
        encoded.append(lang_h.unsqueeze(1).expand(-1, T, -1))

        x = torch.cat(encoded, -1)  # (B, T, H_all)
        B, T, _ = x.shape
        x = x.view(B * T, -1)
        x = self.mlp(x)
        x = x.view(B, T, -1)
        dist = self.policy_head(x)
        return dist

    def get_action(self, data):
        self.eval()
        data = self.preprocess_input(data, train_mode=False)
        with torch.no_grad():
            dist = self.forward(data)
        action = dist.sample().detach().cpu()
        return action.view(action.shape[0], -1).numpy()

    def reset(self):
        pass
