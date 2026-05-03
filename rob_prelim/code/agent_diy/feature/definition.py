#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from common_python.utils.common_func import create_cls
import numpy as np
from agent_diy.conf.conf import Config

# The create_cls function is used to dynamically create a class. The first parameter of the function is the type name,
# and the remaining parameters are the attributes of the class, which should have a default value of None.
# create_cls函数用于动态创建一个类，函数第一个参数为类型名称，剩余参数为类的属性，属性默认值应设为None
ObsData = create_cls(
    "ObsData",
    feature=None,
    legal_act=None,
)


ActData = create_cls(
    "ActData",
    act=None,
)


# SampleData is used to transfer training samples between aisrv and learner.
# SampleData用于在aisrv和learner之间传递训练样本
SampleData = create_cls(
    "SampleData",
    obs=153,  # Observation dimension / 观测维度
    legal_actions=8,  # Legal action dimension / 合法动作维度
    actions=1,  # Action dimension / 动作维度
    probs=8,  # Action probability distribution dimension / 动作概率分布维度
    rewards=1,  # Reward / 奖励
    advantages=1,  # Advantage function / 优势函数
    values=1,  # Value function / 价值函数
    dones=1,  # Whether terminated / 是否结束
)


def reward_shaping(frame_no, score, terminated, truncated, remain_info, _remain_info, obs, _obs):
    """Reward shaping function.

    奖励塑形函数。
    """
    pass


def sample_process(list_game_data):
    """Sample processing function.

    样本处理函数。
    """
    pass
