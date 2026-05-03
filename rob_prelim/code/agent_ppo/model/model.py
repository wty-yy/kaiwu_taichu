#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Neural network model for Robot Vacuum PPO.
清扫大作战 PPO 神经网络模型。
"""

import numpy as np
import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


def init_weights(module, gain=np.sqrt(2)):
    """Orthogonal initialization for Linear/Conv2d modules.

    对 Linear/Conv2d 使用正交初始化。
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class CNN(nn.Module):
    """CNN encoder for 4-channel 21x21 local map."""

    def __init__(self, in_channels=4, output_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            # out: 16x21x21
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            # out: 32x10x10
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0),
            nn.ELU(),
            # out: 64x4x4
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.ELU(),
        )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, output_dim),
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.mlp(self.cnn(x))


class MaskedSetEncoder(nn.Module):
    """Shared-MLP + masked max-pooling encoder for fixed-slot entity sets."""

    def __init__(self, input_dim, max_num, output_dim=64):
        super().__init__()
        self.input_dim = int(input_dim)
        self.max_num = int(max_num)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ELU(),
            nn.Linear(64, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        if x.dim() == 2:
            expected = self.max_num * self.input_dim
            if x.shape[1] != expected:
                raise ValueError(f"set feature shape mismatch, got {x.shape[1]}, expect {expected}")
            x = x.reshape(-1, self.max_num, self.input_dim)
        elif x.dim() == 3:
            if x.shape[1] != self.max_num or x.shape[2] != self.input_dim:
                raise ValueError(
                    f"set feature shape mismatch, got {tuple(x.shape[1:])}, "
                    f"expect {(self.max_num, self.input_dim)}"
                )
        else:
            raise ValueError(f"set feature rank mismatch, got {x.dim()}, expect 2 or 3")

        mask = x.abs().sum(dim=-1) > 0
        shared_features = self.mlp(x)
        shared_features = shared_features.masked_fill(~mask.unsqueeze(-1), -1e9)
        pooled_features, _ = shared_features.max(dim=1)

        has_valid = mask.any(dim=1, keepdim=True)
        pooled_features = torch.where(has_valid, pooled_features, torch.zeros_like(pooled_features))
        return pooled_features


class Model(nn.Module):
    """Set-encoder + map-CNN + actor/critic dual heads."""

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "robot_vacuum_set_cnn"
        self.device = device

        # ===== Feature split aligned with current preprocessor._get_obs =====
        # hero(5) + npc(4 * 19) + charger(4 * 19) + map(4 * 21 * 21)
        self.hero_base_dim = int(getattr(Config, "HERO_BASE_FEATURE_DIM", 5))

        self.npc_item_dim = int(getattr(Config, "NPC_ITEM_FEATURE_DIM", 19))
        self.npc_max_num = int(getattr(Config, "MAX_NPC_ROBOT", 4))
        self.npc_total_dim = self.npc_item_dim * self.npc_max_num

        self.charger_item_dim = int(getattr(Config, "CHARGER_ITEM_FEATURE_DIM", 19))
        self.charger_max_num = int(getattr(Config, "MAX_CHARGER", 4))
        self.charger_total_dim = self.charger_item_dim * self.charger_max_num

        self.map_shape = tuple(getattr(Config, "MAP_FEATURE_SHAPE", (4, 21, 21)))
        if len(self.map_shape) != 3:
            raise ValueError(f"MAP_FEATURE_SHAPE must be (C,H,W), got {self.map_shape}")
        self.map_flat_dim = int(np.prod(self.map_shape))

        self.feature_len = self.hero_base_dim + self.npc_total_dim + self.charger_total_dim + self.map_flat_dim

        # 1) Hero base encoder
        self.hero_encoder = nn.Sequential(
            nn.Linear(self.hero_base_dim, 32),
            nn.ELU(),
            nn.Linear(32, 64),
        )

        # 2) Entity set encoders
        self.npc_encoder = MaskedSetEncoder(
            input_dim=self.npc_item_dim,
            max_num=self.npc_max_num,
            output_dim=64,
        )
        self.charger_encoder = MaskedSetEncoder(
            input_dim=self.charger_item_dim,
            max_num=self.charger_max_num,
            output_dim=64,
        )

        # 3) Map CNN encoder
        self.map_cnn = CNN(in_channels=self.map_shape[0], output_dim=128)

        # 4) Fusion backbone
        self.fusion_mlp = nn.Sequential(
            nn.Linear(64 + 64 + 64 + 128, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )

        # 5) Actor / Critic heads
        self.actor_head = nn.Linear(128, Config.ACTION_NUM)
        self.critic_head = nn.Linear(128, Config.VALUE_NUM)

        self.apply(init_weights)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)

    def forward(self, s, inference=False):
        """Forward pass.

        输入 obs 形状: [B, feature_len]。
        """
        obs = s
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        if obs.shape[1] != self.feature_len:
            raise ValueError(f"obs len mismatch, got {obs.shape[1]}, expect {self.feature_len}")

        i0 = 0
        i1 = i0 + self.hero_base_dim
        i2 = i1 + self.npc_total_dim
        i3 = i2 + self.charger_total_dim
        i4 = i3 + self.map_flat_dim

        hero_x = obs[:, i0:i1]
        npc_x = obs[:, i1:i2]
        charger_x = obs[:, i2:i3]
        map_x = obs[:, i3:i4].reshape(-1, *self.map_shape)

        hero_feat = self.hero_encoder(hero_x)
        npc_feat = self.npc_encoder(npc_x)
        charger_feat = self.charger_encoder(charger_x)
        map_feat = self.map_cnn(map_x)

        fused = torch.cat([hero_feat, npc_feat, charger_feat, map_feat], dim=-1)
        core_feat = self.fusion_mlp(fused)

        return self.actor_head(core_feat), self.critic_head(core_feat)

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
