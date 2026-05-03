#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np


# Configuration
# 配置，包含维度设置，算法参数设置，文件的最后一些配置是开悟平台使用不要改动
class Config:

    # Whether to use CNN networks
    # 是否使用CNN网络
    USE_CNN = False
    VIEW_SIZE = 50 if USE_CNN else 0

    FEATURE_VECTOR_SHAPE = (153,)
    FEATURE_IMAGE_SHAPE = (4, VIEW_SIZE + 1, VIEW_SIZE + 1)

    ACTION_SHAPE = (8,)
    VALUE_SHAPE = (1,)

    # Discount factor GAMMA in RL
    # RL中的回报折扣GAMMA
    GAMMA = 0.95

    # Initial learning rate
    # 初始的学习率
    START_LR = 5e-4

    VALUE_LOSS_COEFF = 0.5
    ENTROPY_LOSS_COEFF = 0.025
