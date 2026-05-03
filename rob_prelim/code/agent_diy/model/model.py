#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Robot Vacuum DIY model implementation.
清扫大作战 DIY 模型实现。
"""


import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    """DIY model class.

    DIY 模型类。
    """

    def __init__(self, state_shape, action_shape=0, softmax=False):
        """Initialize the model.

        初始化模型。
        """
        super().__init__()

        # User-defined network
        # 用户自定义网络
