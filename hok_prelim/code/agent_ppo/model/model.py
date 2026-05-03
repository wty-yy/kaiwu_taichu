#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Neural network model for Gorge Chase PPO.
峡谷追猎 PPO 神经网络模型。
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

PROJECT_CODE_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE_DIR))

from agent_ppo.conf.conf import Config


def _format_size(num_bytes):
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0

def init_weights(module, gain=np.sqrt(2)):
    """
    正交初始化函数
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        # 权重使用正交初始化，增益系数（gain）通常设为 sqrt(2) 配合 ReLU/ELU 激活函数
        nn.init.orthogonal_(module.weight, gain=gain)
        
        # 偏置初始化为 0
        if module.bias is not None:
            nn.init.zeros_(module.bias)

# ---------------- IMPALA 残差块 ----------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.apply(init_weights)

    def forward(self, x):
        out = F.elu(x)
        out = self.conv1(out)
        out = F.elu(out)
        out = self.conv2(out)
        return x + out

# ---------------- IMPALA 卷积组 ----------------
class IMPALABlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 使用 MaxPool 进行下采样，强行保留离散特征（如墙壁边缘、宝箱点位）
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)
        self.apply(init_weights)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

# ---------------- 全新的地图特征提取器 ----------------
class IMPALACNN(nn.Module):
    def __init__(self, in_channels=8, input_size=41, channels=(16, 24, 32), hidden_dim=128, output_dim=96):
        super().__init__()
        c1, c2, c3 = [int(v) for v in channels]
        _ = input_size  # keep constructor signature stable for callers
        self.blocks = nn.Sequential(
            IMPALABlock(in_channels, c1),
            IMPALABlock(c1, c2),
            IMPALABlock(c2, c3),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(init_weights)

    def forward(self, x):
        x = self.blocks(x)
        x = self.pool(x)
        return self.mlp(x)

class MaskedSetEncoder(nn.Module):
    def __init__(self, input_dim, max_num, hidden_dim=32, output_dim=48):
        super().__init__()
        self.input_dim = int(input_dim)
        self.max_num = int(max_num)
        # 针对单个实体的共享 MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim), # 稳定特征分布
        )
        self.apply(init_weights)

    def forward(self, x):
        # x 支持两种输入：
        # 1) [batch_size, max_num * input_dim]
        # 2) [batch_size, max_num, input_dim]
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

        # 有效实体掩码：空槽位默认全 0
        mask = x.abs().sum(dim=-1) > 0
        
        # 1. 独立处理每个实体
        # shared_features: [batch_size, max_num, output_dim]
        shared_features = self.mlp(x) 
        
        # 2. Masking 关键步骤：把空槽位的特征替换为负无穷
        # 扩展 mask 维度对齐特征: [batch_size, max_num, 1]
        mask_expanded = mask.unsqueeze(-1) 
        shared_features = shared_features.masked_fill(~mask_expanded, -1e9)
        
        # 3. Max Pooling (此时负无穷绝不会成为最大值)
        # pooled_features: [batch_size, output_dim]
        pooled_features, _ = shared_features.max(dim=1)

        # 全空集合时回落到 0，避免 -1e9 传播
        has_valid = mask.any(dim=1, keepdim=True)
        pooled_features = torch.where(has_valid, pooled_features, torch.zeros_like(pooled_features))
        
        return pooled_features

class Model(nn.Module):
    """CNN + Actor/Critic dual heads"""

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_lite"
        self.device = device

        self.hero_base_dim = Config.HERO_BASE_FEATURE_DIM

        self.monster_item_dim = sum(Config.MONSTER_FEATURE)
        self.monster_max_num = Config.MAX_MONSTER_NUM
        self.monster_total_dim = Config.MONSTER_FEATURE_DIM

        self.treasure_item_dim = sum(Config.TREASURE_FEATURE)
        self.treasure_max_num = Config.MAX_TREASURE_NUM
        self.treasure_total_dim = Config.TREASURE_FEATURE_DIM

        self.buff_item_dim = sum(Config.BUFF_FEATURE)
        self.buff_max_num = Config.MAX_BUFF_NUM
        self.buff_total_dim = Config.BUFF_FEATURE_DIM

        self.local_obstacle_dim = Config.LOCAL_OBSTACLE_FEATURE_DIM
        self.map_channels = Config.MAP_FEATURE_CHANNELS
        self.map_shape = Config.MAP_FEATURE_SHAPE
        self.map_flat_dim = Config.MAP_FEATURE_DIM
        self.map_cnn_output_dim = Config.MAP_CNN_OUTPUT_DIM
        self.entity_embed_dim = int(Config.ENTITY_EMBED_DIM)
        self.local_obstacle_hidden_dim = int(Config.LOCAL_OBSTACLE_HIDDEN_DIM)
        self.local_obstacle_embed_dim = int(Config.LOCAL_OBSTACLE_EMBED_DIM)
        self.set_encoder_hidden_dim = int(Config.SET_ENCODER_HIDDEN_DIM)
        self.map_cnn_hidden_dim = int(Config.MAP_CNN_HIDDEN_DIM)
        self.fusion_hidden_dim = int(Config.FUSION_HIDDEN_DIM)
        self.fusion_output_dim = int(Config.FUSION_OUTPUT_DIM)

        # 1. 英雄基础特征处理 (升维，与其他特征对齐)
        self.hero_encoder = nn.Sequential(
            nn.Linear(self.hero_base_dim, 32),
            nn.ELU(),
            nn.Linear(32, self.entity_embed_dim),
        )
        
        # 2. 各类集合实体编码器
        self.monster_encoder = MaskedSetEncoder(
            input_dim=self.monster_item_dim,
            max_num=self.monster_max_num,
            hidden_dim=self.set_encoder_hidden_dim,
            output_dim=self.entity_embed_dim,
        )
        self.treasure_encoder = MaskedSetEncoder(
            input_dim=self.treasure_item_dim,
            max_num=self.treasure_max_num,
            hidden_dim=self.set_encoder_hidden_dim,
            output_dim=self.entity_embed_dim,
        )
        self.buff_encoder = MaskedSetEncoder(
            input_dim=self.buff_item_dim,
            max_num=self.buff_max_num,
            hidden_dim=self.set_encoder_hidden_dim,
            output_dim=self.entity_embed_dim,
        )

        # 3. 周围 21x21 障碍物编码器（强调近身防撞信息）
        self.local_obstacle_encoder = nn.Sequential(
            nn.Linear(self.local_obstacle_dim, self.local_obstacle_hidden_dim),
            nn.LayerNorm(self.local_obstacle_hidden_dim),
            nn.ELU(),
            nn.Linear(self.local_obstacle_hidden_dim, self.local_obstacle_embed_dim),
            nn.ELU(),
        )
        
        # 4. 地图 CNN 编码器
        self.map_cnn = IMPALACNN(
            in_channels=self.map_channels,
            input_size=self.map_shape[0],
            channels=Config.MAP_CNN_CHANNELS,
            hidden_dim=self.map_cnn_hidden_dim,
            output_dim=self.map_cnn_output_dim,
        )
        
        # 5. 中央融合层
        fusion_input_dim = self.entity_embed_dim * 4 + self.local_obstacle_embed_dim + self.map_cnn_output_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, self.fusion_hidden_dim),
            nn.LayerNorm(self.fusion_hidden_dim), # 加入 LayerNorm 稳定多模态特征融合
            nn.ELU(),
            nn.Linear(self.fusion_hidden_dim, self.fusion_output_dim),
            nn.ELU(),
        )
        
        # 6. 动作输出头 (Actor) / 价值输出头 (Critic)
        self.actor_head = nn.Linear(self.fusion_output_dim, Config.ACTION_NUM)
        self.critic_head = nn.Linear(self.fusion_output_dim, Config.VALUE_NUM)

        self.apply(init_weights)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)

    def forward(self, obs, inference=False):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        if obs.shape[1] != Config.FEATURE_LEN:
            raise ValueError(f"obs len mismatch, got {obs.shape[1]}, expect {Config.FEATURE_LEN}")

        i0 = 0
        i1 = i0 + self.hero_base_dim
        i2 = i1 + self.monster_total_dim
        i3 = i2 + self.treasure_total_dim
        i4 = i3 + self.buff_total_dim
        i5 = i4 + self.local_obstacle_dim
        i6 = i5 + self.map_flat_dim

        hero_x = obs[:, i0:i1]
        monster_x = obs[:, i1:i2]
        treasure_x = obs[:, i2:i3]
        buff_x = obs[:, i3:i4]
        local_obstacle_x = obs[:, i4:i5]
        packed_map = obs[:, i5:i6].reshape(-1, *self.map_shape).long()

        # 位运算解压
        terrain = packed_map & 0b00000011                  # 提取 0-1 位
        visited = (packed_map >> 2) & 0b00000011           # 提取 2-3 位
        monster = (packed_map >> 4) & 0b00000011           # 提取 4-5 位
        treasure = (packed_map >> 6) & 0b00000001          # 提取 6 位
        buff = (packed_map >> 7) & 0b00000001              # 提取 7 位

        # 恢复为 8 通道的 float32 Tensor，完美契合神经网络的输入形式
        ch0 = (terrain == 0).float()
        ch1 = (terrain == 1).float()
        ch2 = (terrain == 2).float()
        ch3 = (terrain == 3).float()
        
        # 恢复原有的 0.0, 0.5, 1.0 的连续数值
        ch4 = visited.float() * 0.5 
        ch5 = monster.float() * 0.5

        ch6 = treasure.float()
        ch7 = buff.float()

        map_x = torch.stack([ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7], dim=1)

        # 正常喂给 IMPALA CNN

        hero_feat = self.hero_encoder(hero_x)
        monster_feat = self.monster_encoder(monster_x)
        treasure_feat = self.treasure_encoder(treasure_x)
        buff_feat = self.buff_encoder(buff_x)
        local_obstacle_feat = self.local_obstacle_encoder(local_obstacle_x)
        map_feat = self.map_cnn(map_x)
        
        # 将所有提炼后的特征拼接
        fused = torch.cat([
            hero_feat, monster_feat, treasure_feat, buff_feat, local_obstacle_feat, map_feat
        ], dim=-1)
        
        core_feat = self.fusion_mlp(fused)
        
        return self.actor_head(core_feat), self.critic_head(core_feat)

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()


if __name__ == "__main__":
    model = Model()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected_ckpt_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())

    print("=== Model Summary ===")
    print(model)
    print()
    print("=== Parameter Stats ===")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Expected FP32 checkpoint size: {_format_size(expected_ckpt_bytes)}")
