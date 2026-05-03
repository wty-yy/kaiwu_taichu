#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Gorge Chase PPO.
峡谷追猎 PPO 训练工作流。
"""

import os
import time

import numpy as np
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import SampleData, sample_process
from agent_ppo.workflow.map_sampler import WeightedMapSampler
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    # Read user config / 读取用户配置
    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0
        self.map_sampler = WeightedMapSampler(
            usr_conf=self.usr_conf,
            map_sample_weights=Config.MAP_SAMPLE_WEIGHTS,
            default_weight=Config.MAP_SAMPLE_DEFAULT_WEIGHT,
            logger=self.logger,
        )

    def run_episodes(self):
        """Run a single episode and yield collected samples.

        执行单局对局并 yield 训练样本。
        """
        while True:
            # Periodically fetch training metrics / 定期获取训练指标
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            # Reset env / 重置环境
            self.map_sampler.prepare_for_reset()
            env_obs = self.env.reset(self.usr_conf)

            # Disaster recovery / 容灾处理
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            # Reset agent & load latest model / 重置 Agent 并加载最新模型
            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            # Initial observation / 初始观测处理
            obs_data, remain_info = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0
            episode_reward_breakdown = {
                "reward_alive": 0.0,
                "reward_treasure": 0.0,
                "reward_buff": 0.0,
                "reward_approach": 0.0,
                "reward_escape_danger": 0.0,
                "reward_treasure_danger": 0.0,
                "reward_invalid_move": 0.0,
                "reward_bad_flash": 0.0,
                "reward_explore_map": 0.0,
                "reward_repeat": 0.0,
                "reward_procrastination": 0.0,
                "reward_terminal": 0.0,
            }

            self.logger.info(f"Episode {self.episode_cnt} start")

            while not done:
                # Predict action / Agent 推理（随机采样）
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)

                # Step env / 与环境交互
                env_reward, env_obs = self.env.step(act)

                # Disaster recovery / 容灾处理
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                step += 1
                done = terminated or truncated

                # Next observation / 处理下一步观测
                _obs_data, _remain_info = self.agent.observation_process(env_obs)

                # Step reward / 每步即时奖励
                reward = np.array(_remain_info.get("reward", [0.0]), dtype=np.float32)
                total_reward += float(reward[0])
                step_reward_breakdown = _remain_info.get("reward_breakdown", {})
                for key in episode_reward_breakdown:
                    if key == "reward_terminal":
                        continue
                    episode_reward_breakdown[key] += float(step_reward_breakdown.get(key, 0.0))

                # Terminal reward / 终局奖励
                final_reward = np.zeros(1, dtype=np.float32)
                if done:
                    env_info = env_obs["observation"]["env_info"]
                    total_score = env_info.get("total_score", 0)

                    if terminated:
                        final_reward[0] = Config.REW_FAIL * Config.REW_GLOBAL_SCALE
                        result_str = "FAIL"
                    else:
                        final_reward[0] = Config.REW_WIN * Config.REW_GLOBAL_SCALE
                        result_str = "WIN"
                    episode_reward_breakdown["reward_terminal"] = float(final_reward[0])

                    self.logger.info(
                        f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} "
                        f"result:{result_str} sim_score:{total_score:.1f} "
                        f"total_reward:{total_reward:.3f}"
                    )

                # Build sample frame / 构造样本帧
                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.float32),
                    reward=reward,
                    done=np.array([float(done)], dtype=np.float32),
                    reward_sum=np.zeros(1, dtype=np.float32),
                    value=np.array(act_data.value, dtype=np.float32).flatten()[:1],
                    next_value=np.zeros(1, dtype=np.float32),
                    advantage=np.zeros(1, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                )
                collector.append(frame)

                # Episode end / 对局结束
                if done:
                    if collector:
                        collector[-1].reward = collector[-1].reward + final_reward

                    # Monitor report / 监控上报
                    now = time.time()
                    if self.monitor:
                        monitor_data = {
                            "reward": round(total_reward + float(final_reward[0]), 4),
                            "episode_steps": step,
                            "episode_cnt": self.episode_cnt,
                            "reward_alive": round(episode_reward_breakdown["reward_alive"], 4),
                            "reward_treasure": round(episode_reward_breakdown["reward_treasure"], 4),
                            "reward_buff": round(episode_reward_breakdown["reward_buff"], 4),
                            "reward_approach": round(episode_reward_breakdown["reward_approach"], 4),
                            "reward_escape_danger": round(episode_reward_breakdown["reward_escape_danger"], 4),
                            "reward_treasure_danger": round(episode_reward_breakdown["reward_treasure_danger"], 4),
                            "reward_invalid_move": round(episode_reward_breakdown["reward_invalid_move"], 4),
                            "reward_bad_flash": round(episode_reward_breakdown["reward_bad_flash"], 4),
                            "reward_explore_map": round(episode_reward_breakdown["reward_explore_map"], 4),
                            "reward_repeat": round(episode_reward_breakdown["reward_repeat"], 4),
                            "reward_procrastination": round(episode_reward_breakdown["reward_procrastination"], 4),
                            "reward_terminal": round(episode_reward_breakdown["reward_terminal"], 4),
                            "n_hit_wall": int(_remain_info.get("n_hit_wall", 0)),
                        }
                        sample_counts = self.map_sampler.get_sample_counts()
                        for map_id in Config.MAP_SAMPLE_MONITOR_IDS:
                            monitor_data[f"map_sample_count_{int(map_id)}"] = int(
                                sample_counts.get(int(map_id), 0)
                            )
                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now

                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                # Update state / 状态更新
                obs_data = _obs_data
                remain_info = _remain_info
