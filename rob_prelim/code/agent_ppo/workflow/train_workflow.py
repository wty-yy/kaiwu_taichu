#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Robot Vacuum.
清扫大作战训练工作流。
"""

import os
import time

import numpy as np

from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import SampleData, sample_process
from agent_ppo.workflow.data_augmentation import RotationDataAugmentation
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    # Read and validate user configuration
    # 读取和校验用户配置
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
        self.data_augmentation = self._build_data_augmentation()

    def _build_data_augmentation(self):
        use_data_augmentation = bool(Config.USE_DATA_AUGMENTATION)
        rotation_steps = Config.AUGMENT_ROTATION_STEPS
        if not use_data_augmentation:
            return None
        self.logger.info(f"data augmentation enabled, rotation_steps={rotation_steps}")
        return RotationDataAugmentation(rotation_steps=rotation_steps, logger=self.logger)

    def run_episodes(self):
        """Run a single episode and yield collected samples.

        单局流程（generator），完成一局后 yield 整局样本。
        """
        while True:
            # Periodically get training metrics
            # 定期打印训练指标
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics: {training_metrics}")

            # Reset environment
            # 重置环境
            env_obs = self.env.reset(self.usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            # Reset agent and load latest model
            # 重置 Agent，加载最新模型
            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            # Initial observation processing
            # 初始观测
            obs_data, _ = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            episode_rotation_step = 0
            if self.data_augmentation is not None:
                episode_rotation_step = self.data_augmentation.sample_rotation_step()
                self.logger.info(f"Episode {self.episode_cnt} rotation_step={episode_rotation_step}")
            done = False
            step = 0
            total_reward = 0.0
            episode_reward_breakdown = {
                "reward_valid_clean": 0.0,
                "reward_edge_clean": 0.0,
                "reward_continuous_clean": 0.0,
                "reward_continuous_no_clean": 0.0,
                "reward_step_punish": 0.0,
                "reward_no_move": 0.0,
                "reward_danger_distance": 0.0,
                "reward_approach_charger_low_battery": 0.0,
                "reward_terminal": 0.0,
            }

            self.logger.info(f"Episode {self.episode_cnt} start")

            while not done:
                # Agent inference / 推理动作
                obs_data_for_policy = obs_data
                if self.data_augmentation is not None:
                    obs_data_for_policy = self.data_augmentation.rotate_obs_data(obs_data, episode_rotation_step)

                act_data_list = self.agent.predict([obs_data_for_policy])
                act_data = act_data_list[0]
                act = int(act_data.action[0])
                if self.data_augmentation is not None:
                    act = self.data_augmentation.inverse_rotate_action_index(act, episode_rotation_step)
                self.agent.last_action = act

                # Environment step / 与环境交互
                _, env_obs = self.env.step(act)
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                frame_no = env_obs["frame_no"]
                step += 1
                done = terminated or truncated

                # Process next observation
                # 特征处理
                _obs_data, _ = self.agent.observation_process(env_obs)
                _obs_data.frame_no = frame_no

                reward_scalar = float(self.agent.last_reward)
                total_reward += reward_scalar

                step_reward_breakdown = self.agent.preprocessor.state_manager.last_reward_breakdown
                episode_reward_breakdown["reward_valid_clean"] += float(step_reward_breakdown["valid_clean"])
                episode_reward_breakdown["reward_edge_clean"] += float(step_reward_breakdown["edge_clean"])
                episode_reward_breakdown["reward_continuous_clean"] += float(step_reward_breakdown["continuous_clean"])
                episode_reward_breakdown["reward_continuous_no_clean"] += float(
                    step_reward_breakdown["continuous_no_clean"]
                )
                episode_reward_breakdown["reward_step_punish"] += float(step_reward_breakdown["step_punish"])
                episode_reward_breakdown["reward_no_move"] += float(step_reward_breakdown["no_move"])
                episode_reward_breakdown["reward_danger_distance"] += float(step_reward_breakdown["danger_distance"])
                episode_reward_breakdown["reward_approach_charger_low_battery"] += float(
                    step_reward_breakdown["approach_charger_low_battery"]
                )

                # Terminal reward calculation
                # 终局奖励
                final_reward = 0.0
                end_by_npc_collision = 0.0
                end_by_battery_depleted = 0.0
                if done:
                    if truncated:
                        # 达到最大步数不额外给奖
                        final_reward = 0.0
                        result_str = "WIN"
                    else:
                        # 提前结束：当前电量非零视为撞到 NPC，否则视为电量耗尽
                        current_battery = int(env_obs["observation"]["frame_state"]["heroes"]["battery"])
                        if current_battery > 0:
                            final_reward = float(Config.REW_NPC_COLLISION_TERMINATE) * Config.REW_GLOBAL_SCALE
                            end_by_npc_collision = 1.0
                        else:
                            final_reward = float(Config.REW_BATTERY_DEPLETED_TERMINATE) * Config.REW_GLOBAL_SCALE
                            end_by_battery_depleted = 1.0
                        result_str = "FAIL"

                    self.logger.info(
                        f"[GAMEOVER] ep:{self.episode_cnt} steps:{step} "
                        f"result:{result_str} final_bonus:{final_reward:.2f} "
                        f"total_reward:{total_reward:.3f} "
                    )
                    episode_reward_breakdown["reward_terminal"] = float(final_reward)

                # Build sample frame
                # 构造样本帧
                reward_arr = np.array([reward_scalar], dtype=np.float32)
                value_arr = act_data.value.flatten()[: Config.VALUE_NUM]

                frame = SampleData(
                    obs=np.array(obs_data_for_policy.feature, dtype=np.float32),
                    legal_action=np.array(obs_data_for_policy.legal_action, dtype=np.float32),
                    act=np.array(act_data.action),
                    reward=reward_arr,
                    done=np.array([float(done)]),
                    reward_sum=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    value=value_arr,
                    next_value=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    advantage=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                )
                collector.append(frame)

                if done:
                    # Add terminal reward to last frame
                    # 终局奖励叠加到最后一步
                    collector[-1].reward = collector[-1].reward + np.array([final_reward], dtype=np.float32)

                    # Monitor reporting / 监控上报
                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        self.monitor.put_data(
                            {
                                os.getpid(): {
                                    "reward": total_reward + final_reward,
                                    "episode_cnt": self.episode_cnt,
                                    "final_reward": float(final_reward),
                                    "end_by_npc_collision": float(end_by_npc_collision),
                                    "end_by_battery_depleted": float(end_by_battery_depleted),
                                    "reward_valid_clean": round(episode_reward_breakdown["reward_valid_clean"], 4),
                                    "reward_edge_clean": round(episode_reward_breakdown["reward_edge_clean"], 4),
                                    "reward_continuous_clean": round(
                                        episode_reward_breakdown["reward_continuous_clean"], 4
                                    ),
                                    "reward_continuous_no_clean": round(
                                        episode_reward_breakdown["reward_continuous_no_clean"], 4
                                    ),
                                    "reward_step_punish": round(episode_reward_breakdown["reward_step_punish"], 4),
                                    "reward_no_move": round(episode_reward_breakdown["reward_no_move"], 4),
                                    "reward_danger_distance": round(episode_reward_breakdown["reward_danger_distance"], 4),
                                    "reward_approach_charger_low_battery": round(
                                        episode_reward_breakdown["reward_approach_charger_low_battery"], 4
                                    ),
                                    "reward_terminal": round(episode_reward_breakdown["reward_terminal"], 4),
                                }
                            }
                        )
                        self.last_report_monitor_time = now

                    # Compute GAE and yield samples
                    # GAE 计算并 yield 样本
                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                # Advance state / 状态推进
                obs_data = _obs_data
