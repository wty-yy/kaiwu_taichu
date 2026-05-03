#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Robot Vacuum PPO agent.
清扫大作战 PPO 配置。
"""


class Config:

    # Observation feature dimensions aligned with preprocessor/model:
    # 7(hero) + 4*19(npc set) + 4*19(charger set) + 4*21*21(map 4 channels) = 1923
    # 与 preprocessor/model 对齐的观测特征维度：1923
    HERO_BASE_FEATURE_DIM = 7
    NPC_ITEM_FEATURE_DIM = 19
    MAX_NPC_ROBOT = 4
    CHARGER_ITEM_FEATURE_DIM = 19
    MAX_CHARGER = 4
    MAP_FEATURE_SHAPE = (4, 21, 21)

    FEATURES = [
        HERO_BASE_FEATURE_DIM,
        MAX_NPC_ROBOT * NPC_ITEM_FEATURE_DIM,
        MAX_CHARGER * CHARGER_ITEM_FEATURE_DIM,
        MAP_FEATURE_SHAPE[0] * MAP_FEATURE_SHAPE[1] * MAP_FEATURE_SHAPE[2],
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURES)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space: 8 directional moves
    # 动作空间：8个方向移动
    ACTION_NUM = 8

    # Single-head value
    # 单头价值
    VALUE_NUM = 1

    # ========== PPO hyperparameters / PPO 超参数 ==========
    GAMMA = 0.99
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.00035
    INIT_LEARNING_RATE_END = 0.00025
    LR_STEPS = 50000

    # 熵系数线性衰减（鼓励前期探索，后期收敛）
    BETA_START = 0.01
    BETA_END = 0.0001
    BETA_STEPS = 100000

    CLIP_PARAM = 0.2
    VF_COEF = 0.5

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5
    # ==================================================

    #################### 奖励模板 ####################
    # 有效清洁奖励（按本步清洁格数累计）
    REW_VALID_CLEAN: float = 0.1
    # 连续清扫奖励系数（清扫历史 buffer 已满且窗内每步清扫格数均>0时给予该分量）
    REW_CONTINUOUS_CLEAN: float = 0.05
    # 连续未清扫惩罚系数（未清扫标记 buffer 已满且窗内每步均为未清扫时给予该分量，应为负数）
    REW_CONTINUOUS_NO_CLEAN: float = -0.1
    # 清扫历史 buffer 最大长度（两 buffer 共用；满窗时按「窗内是否全为清扫/全为未清扫」判定连续类奖惩）
    CLEAN_HISTORY_BUFFER_MAXLEN: int = 3
    # 撞到 NPC 终止惩罚
    REW_NPC_COLLISION_TERMINATE: float = -20.0
    # 电量耗尽终止惩罚
    REW_BATTERY_DEPLETED_TERMINATE: float = -10.0
    # 每步惩罚
    REW_STEP_PUNISH: float = -0.01
    # 不移动惩罚
    REW_NO_MOVE: float = -0.2
    # 危险距离参数: -max(REW_DANGER_RADIUS - 最近NPC距离, 0) * REW_DANGER_FACTOR
    REW_DANGER_RADIUS: float = 5.0
    REW_DANGER_FACTOR: float = 0.1
    # 低电量靠近充电桩奖励系数
    REW_LOW_BATTERY_APPROACH_CHARGER: float = 0.05
    # 低电量阈值插值表（16个锚点，对应 dist=0,5,10,...,75格，用 np.interp 按精确距离插值）
    # 锚点设计：dist=0 时 0.50（近桩清扫自由最大），dist≈37 时 ≈0.60（对齐 v2.4.1 有效值），
    #           dist=75 时 0.70（最远上限，比原 1.0 大幅降低，杜绝全程低电量模式）
    LOW_BATTERY_THRESHOLD_INTERP = [
        0.50, 0.51, 0.53, 0.54, 0.55, 0.57,   # dist=0,5,10,15,20,25
        0.58, 0.59, 0.61, 0.62, 0.63, 0.65,   # dist=30,35,40,45,50,55
        0.66, 0.67, 0.69, 0.70,               # dist=60,65,70,75
    ]
    # 边缘清扫系数插值表（16个锚点，对应 dist=0,5,10,...,75格）
    # 锚点设计：dist=0 时 0.050（= v2.4 原值），dist≈37 时 ≈0.030（对齐 v2.4.1 有效值），
    #           dist=75 时 0.010（仍有正向信号，原 bin 3+ 只有 0.005）
    REW_EDGE_CLEAN_INTERP = [
        0.050, 0.047, 0.044, 0.042, 0.039, 0.036,  # dist=0,5,10,15,20,25
        0.033, 0.031, 0.028, 0.025, 0.022, 0.019,  # dist=30,35,40,45,50,55
        0.017, 0.014, 0.011, 0.010,                # dist=60,65,70,75
    ]
    # 全局奖励缩放系数
    REW_GLOBAL_SCALE: float = 0.1

    # ========== Data augmentation / 数据增强 ==========
    # 是否启用训练样本旋转增强
    USE_DATA_AUGMENTATION = True
    # 顺时针旋转步数：0/1/2/3 -> 0/90/180/270 度
    AUGMENT_ROTATION_STEPS = [0, 1, 2, 3]
