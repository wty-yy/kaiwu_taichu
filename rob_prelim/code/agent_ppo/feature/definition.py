#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Data definition and GAE computation for Robot Vacuum.
清扫大作战数据类定义与 GAE 计算。
"""

import numpy as np
from common_python.utils.common_func import create_cls
from agent_ppo.conf.conf import Config


# ObsData: feature vector + legal action mask
# 观测数据：feature 为特征向量，legal_action 为合法动作掩码
ObsData = create_cls("ObsData", feature=None, legal_action=None)

# ActData: sampled action, greedy action, action probabilities, state value
# 动作数据：action 为采样动作，d_action 为贪心动作，prob 为动作概率，value 为状态价值
ActData = create_cls(
    "ActData",
    action=None,
    d_action=None,
    prob=None,
    value=None,
)

# SampleData: int values are treated as dimensions by the framework
# 训练样本数据：字段值为 int 时框架自动按维度处理
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,  # feature vector / 特征向量
    legal_action=Config.ACTION_NUM,  # 8D legal action mask / 合法动作掩码
    act=1,  # action index / 执行的动作
    reward=Config.VALUE_NUM,  # 1D reward / 奖励
    reward_sum=Config.VALUE_NUM,  # GAE td-lambda return
    done=1,
    value=Config.VALUE_NUM,  # 1D value estimate / 价值估计
    next_value=Config.VALUE_NUM,
    advantage=Config.VALUE_NUM,  # 1D GAE advantage / GAE 优势
    prob=Config.ACTION_NUM,  # 8D action probabilities / 动作概率
)


def sample_process(list_sample_data):
    """Fill next_value and compute GAE advantage.

    计算 GAE 并填充 next_value。
    """
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value

    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data):
    """Compute advantage and cumulative return using GAE(λ).

    使用 GAE(λ) 计算优势函数与累积回报。
    """
    gae = 0.0
    gamma = Config.GAMMA
    lamda = Config.LAMDA
    for sample in reversed(list_sample_data):
        delta = -sample.value + sample.reward + gamma * sample.next_value
        gae = gae * gamma * lamda + delta
        sample.advantage = gae
        sample.reward_sum = gae + sample.value
