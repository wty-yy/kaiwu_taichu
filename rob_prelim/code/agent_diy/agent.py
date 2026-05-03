#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Robot Vacuum DIY Agent class based on kaiwudrl BaseAgent interface.
清扫大作战 DIY Agent 主类，基于 kaiwudrl BaseAgent 接口。
"""


import torch
from kaiwudrl.interface.agent import BaseAgent
from agent_diy.model.model import Model
from agent_diy.conf.conf import Config


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        """Initialize the agent.

        初始化 Agent。
        """
        super().__init__(agent_type, device, logger, monitor)

    def predict(self, list_obs_data):
        """Predict action from observation data.

        根据观测数据推理动作。
        """
        pass

    def exploit(self, list_obs_data):
        """Evaluation mode inference (greedy).

        评估模式推理（贪心）。
        """
        pass

    def learn(self, list_sample_data):
        """Train the model.

        训练模型。
        """
        pass

    def save_model(self, path=None, id="1"):
        """Save model checkpoint.

        保存模型检查点。
        """
        pass

    def load_model(self, path=None, id="1"):
        """Load model checkpoint.

        加载模型检查点。
        """
        pass

    def observation_process(self, obs, preprocessor, extra_info=None):
        """
        This function is an important feature processing function, mainly responsible for:
            - Parsing information in the raw data
            - Parsing preprocessed feature data
            - Processing the features and returning the processed feature vector
            - Concatenation of features
            - Annotation of legal actions
        Function inputs:
            - obs: Local observation information returned by the environment
            - preprocessor: Preprocessor
            - extra_info: Global information returned by the environment
        Function outputs:
            - ObsData: Observation data for model inference
            - remain_info: Other data for reward calculation

        该函数是特征处理的重要函数, 主要负责：
            - 解析原始数据里的信息
            - 解析预处理后的特征数据
            - 对特征进行处理, 并返回处理后的特征向量
            - 特征的拼接
            - 合法动作的标注
        函数的输入：
            - obs: 环境返回的局部观测信息
            - preprocessor: 预处理器
            - extra_info: 环境返回的全局状态信息
        函数的输出：
            - ObsData: 用于模型推理的观测数据
            - remain_info: 用于奖励计算的其他数据
        """
        pass

    def action_process(self, act_data):
        pass
