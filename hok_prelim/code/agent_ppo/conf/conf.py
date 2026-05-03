#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Gorge Chase PPO.
峡谷追猎 PPO 配置。
"""

import numpy as np


class Config:

    # ===== Map sampling control / 地图采样控制（在 train_workflow 中生效） =====
    MAP_SAMPLE_DEFAULT_WEIGHT = 1.0                                # 未显式配置地图的默认采样权重
    MAP_SAMPLE_WEIGHTS = {
        1: 2.0,
        2: 2.0,
    }                                                              # 地图采样权重；其余地图使用默认权重
    MAP_SAMPLE_MONITOR_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]       # 监控面板展示的地图采样计数 map_id 列表

    # ===== Observation dimensions / 观测维度（与 StateManager.get_obs 对齐） =====
    HERO_BASE_FEATURE = [
        2,  # 英雄坐标 (x, z)
        1,  # 剩余宝箱占比
        1,  # 步数进度
        1,  # 闪现冷却归一化
        1,  # buff 持续时间归一化
    ]
    HERO_BASE_FEATURE_DIM = sum(HERO_BASE_FEATURE)                      # 英雄基础特征总维度（6）

    MONSTER_FEATURE = [
        2,  # 怪物坐标 (x, z)
        6,  # 与英雄距离桶 one-hot
        9,  # 相对方位 one-hot
        1,  # 怪物速度归一化
        1,  # 怪物是否在视野内
    ]
    MAX_MONSTER_NUM = 2                                                 # 预留怪物槽位数
    MONSTER_FEATURE_DIM = sum(MONSTER_FEATURE) * MAX_MONSTER_NUM        # 怪物特征总维度（38）

    TREASURE_FEATURE = [
        2,  # 宝箱坐标 (x, z)
        6,  # 与英雄距离桶 one-hot
        9,  # 相对方位 one-hot
    ]
    MAX_TREASURE_NUM = 10                                                # 预留宝箱槽位数
    TREASURE_FEATURE_DIM = sum(TREASURE_FEATURE) * MAX_TREASURE_NUM      # 宝箱特征总维度（170）

    BUFF_FEATURE = [
        2,  # buff 坐标 (x, z)
        6,  # 与英雄距离桶 one-hot
        9,  # 相对方位 one-hot
    ]
    MAX_BUFF_NUM = 2                                                     # 预留 buff 槽位数
    BUFF_FEATURE_DIM = sum(BUFF_FEATURE) * MAX_BUFF_NUM                  # buff 特征总维度（34）

    MAP_INFO_SHAPE = (21, 21)                                            # 局部障碍地图尺寸（环境输入）
    LOCAL_OBSTACLE_FEATURE_DIM = int(np.prod(MAP_INFO_SHAPE))            # 周围 21x21 障碍物特征维度
    MAP_FEATURE_SIZE = 41                                                # 地图特征分支窗口大小（以智能体为中心，奇数保证英雄位于唯一中心格）
    MAP_FEATURE_CHANNELS = 8                                             # 解压后地图特征通道数
    MAP_FEATURE_SHAPE = (MAP_FEATURE_SIZE, MAP_FEATURE_SIZE)             # 压缩地图分支输入形状（41x41）
    MAP_FEATURE_DIM = int(np.prod(MAP_FEATURE_SHAPE))                    # 地图分支展平维度（1681）
    ENTITY_EMBED_DIM = 48                                                # 英雄/怪物/宝箱/buff 编码维度
    LOCAL_OBSTACLE_HIDDEN_DIM = 128                                      # 周围障碍物 MLP 隐层维度
    LOCAL_OBSTACLE_EMBED_DIM = 48                                        # 周围障碍物编码维度
    SET_ENCODER_HIDDEN_DIM = 32                                          # 集合实体共享 MLP 隐层维度
    MAP_CNN_CHANNELS = (20, 32, 48)                                      # 地图 CNN 各阶段通道数，适当增大以提升地图表达能力
    MAP_CNN_HIDDEN_DIM = 256                                             # 地图 CNN 池化后的隐层维度
    MAP_CNN_OUTPUT_DIM = 192                                             # 地图 CNN 编码输出维度
    FUSION_HIDDEN_DIM = 320                                              # 多模态融合层第一层宽度
    FUSION_OUTPUT_DIM = 160                                              # 多模态融合后的核心特征维度

    HERO_FEATURE = [
        HERO_BASE_FEATURE_DIM,  # 英雄基础特征
        MONSTER_FEATURE_DIM,    # 怪物特征
        TREASURE_FEATURE_DIM,   # 宝箱特征
        BUFF_FEATURE_DIM,       # buff 特征
    ]
    HERO_FEATURE_DIM = sum(HERO_FEATURE)                                 # 前置向量特征总维度（248）
    FEATURES = HERO_FEATURE + [LOCAL_OBSTACLE_FEATURE_DIM, MAP_FEATURE_DIM]  # 全部特征分段维度
    FEATURE_SPLIT_SHAPE = FEATURES                                       # 与 FEATURES 保持一致
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)                               # 观测总维度（2370）
    DIM_OF_OBSERVATION = FEATURE_LEN                                     # 样本中的 obs 维度

    # Action space / 动作空间：8个移动方向，后8个对应闪现方向
    ACTION_NUM = 16

    # Value head / 价值头：单头生存奖励
    VALUE_NUM = 1

    # ========== PPO hyperparameters / PPO 超参数 ==========
    GAMMA = 0.99
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.00035
    INIT_LEARNING_RATE_END = 0.00025
    LR_STEPS = 50000

    # 熵系数线性衰减（鼓励前期探索，后期收敛）
    BETA_START = 0.01
    BETA_END = 0.001
    BETA_STEPS = 50000

    CLIP_PARAM = 0.2
    VF_COEF = 0.5
    GRAD_CLIP_RANGE = 0.5
    # ==================================================

    # =========== Reward configuration / 奖励配置 ===========
    # 奖励配置（与 StateManager.get_reward 对齐）
    REW_ALIVE = 0.05                 # 每步存活奖励
    REW_TREASURE = 5.0               # 宝箱获取奖励
    REW_BUFF = 3.0                   # Buff 获取奖励
    REW_APPROACH = 0.01              # 靠近最近宝箱塑形（仅在附近怪物风险可接受时生效）
    APPROACH_SAFE_MONSTER_DISTANCE = 5.0  # 靠近宝箱奖励的安全阈值：最近可见怪物距离至少达到该值才给奖励
    REW_ESCAPE_DANGER = 0.08         # 危险区内成功拉开与最近怪物距离的奖励
    REW_TREASURE_DANGER = -0.2       # 贴近怪物时拿宝箱的额外风险惩罚
    REW_INVALID_MOVE = -0.5          # 无效移动惩罚（未放技能撞墙）
    REW_BAD_FLASH = -2.0             # 闪现惩罚：仅满足豁免条件时不惩罚，否则固定 -5
    REW_EXPLORE_MAP = 1.0 / 2000     # 新开未探索可通过区域奖励系数
    REW_REPEAT_SUM_THRESHOLD = 25.0  # 重复经过惩罚阈值（5x5历史步数和）
    REW_REPEAT_SCALE = -0.06         # 重复经过惩罚系数：避免必要绕怪/折返被过度惩罚
    PROCRASTINATION_WINDOW_STEPS = 15      # 拖延判定窗口步数
    PROCRASTINATION_MIN_MOVE_DISTANCE = 6.0  # 拖延判定最小位移阈值
    REW_PROCRASTINATION = -0.1       # 拖延惩罚：窗口内位移不足阈值时触发；拿到宝箱或 buff 会重置窗口
    REW_GLOBAL_SCALE = 0.1           # 全局缩放

    # 终局奖励
    REW_FAIL = -20.0                 # 终局失败奖励
    REW_WIN = 0.0                    # 终局通关奖励
    # ==================================================
