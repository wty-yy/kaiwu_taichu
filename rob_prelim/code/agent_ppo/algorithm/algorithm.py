#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Standard PPO algorithm for Robot Vacuum.
清扫大作战 PPO 算法。

Loss composition / 损失组成：
  total_loss = vf_coef * value_loss + policy_loss - beta * entropy_loss
"""

import os
import time

import torch

from agent_ppo.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.model = model
        self.optimizer = optimizer
        self.parameters = [p for pg in optimizer.param_groups for p in pg["params"]]
        self.device = device
        self.logger = logger
        self.monitor = monitor

        self.clip_param = Config.CLIP_PARAM
        self.vf_coef = Config.VF_COEF

        # Dynamic schedules for learning rate and entropy coefficient.
        self.lr_start = float(Config.INIT_LEARNING_RATE_START)
        self.lr_end = float(getattr(Config, "INIT_LEARNING_RATE_END", Config.INIT_LEARNING_RATE_START))
        self.lr_steps = int(getattr(Config, "LR_STEPS", 0))

        self.beta_start = float(Config.BETA_START)
        self.beta_end = float(getattr(Config, "BETA_END", Config.BETA_START))
        self.beta_steps = int(getattr(Config, "BETA_STEPS", 0))

        self.current_lr = self.lr_start
        self.var_beta = self.beta_start
        self.label_size = Config.ACTION_NUM

        self.train_step = 0
        self.first_load = True
        self.last_report_time = 0

    def learn(self, list_sample_data):
        """Training entry: perform one PPO gradient step on a batch of SampleData.

        训练入口：接收一批 SampleData，执行一步梯度更新。
        """
        # Update LR/entropy schedules before this optimization step.
        self._update_schedules()

        obs = torch.stack([s.obs for s in list_sample_data]).to(self.device)
        legal_action = torch.stack([s.legal_action for s in list_sample_data]).to(self.device)
        act = torch.stack([s.act for s in list_sample_data]).to(self.device).view(-1, 1)
        old_prob = torch.stack([s.prob for s in list_sample_data]).to(self.device)
        old_value = torch.stack([s.value for s in list_sample_data]).to(self.device)
        reward_sum = torch.stack([s.reward_sum for s in list_sample_data]).to(self.device)
        advantage = torch.stack([s.advantage for s in list_sample_data]).to(self.device)
        reward = torch.stack([s.reward for s in list_sample_data]).to(self.device)

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        rst_list = self.model(obs)
        logits, value_pred = rst_list[0], rst_list[1]
        rnd_pred, rnd_target = None, None
        if bool(getattr(Config, "USE_RND_NETWORK", False)) and len(rst_list) >= 4:
            rnd_pred, rnd_target = rst_list[2], rst_list[3]

        total_loss, info = self._compute_loss(
            logits=logits,
            value_pred=value_pred,
            legal_action=legal_action,
            old_action=act,
            old_prob=old_prob,
            old_value=old_value,
            reward_sum=reward_sum,
            advantage=advantage,
            rnd_pred=rnd_pred,
            rnd_target=rnd_target,
        )

        total_loss.backward()

        if Config.USE_GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)

        self.optimizer.step()
        self.train_step += 1

        results = {
            "total_loss": total_loss.item(),
            "learning_rate": self.current_lr,
            "ent_coef": self.var_beta,
        }

        # Periodic monitoring report
        # 定期上报监控
        now = time.time()
        if now - self.last_report_time >= 60:
            results["value_loss"] = round(info["value_loss"], 4)
            results["policy_loss"] = round(info["policy_loss"], 4)
            results["entropy_loss"] = round(info["entropy_loss"], 4)
            if "rnd_loss" in info:
                results["rnd_loss"] = round(info["rnd_loss"], 4)
            results["reward"] = round(reward.mean().item(), 4)
            results["learning_rate"] = round(self.current_lr, 8)
            results["ent_coef"] = round(self.var_beta, 8)

            self.logger.info(
                f"policy_loss: {results['policy_loss']}, "
                f"value_loss: {results['value_loss']}, "
                f"entropy_loss: {results['entropy_loss']}, "
                f"lr: {results['learning_rate']}, "
                f"ent_coef: {results['ent_coef']}"
                + (f", rnd_loss: {results['rnd_loss']}" if "rnd_loss" in results else "")
            )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})

            self.last_report_time = now

        return results

    def _compute_loss(
        self,
        logits,
        value_pred,
        legal_action,
        old_action,
        old_prob,
        old_value,
        reward_sum,
        advantage,
        rnd_pred=None,
        rnd_target=None,
    ):
        """Compute standard PPO loss (policy + value + entropy).

        计算标准 PPO 三项损失。
        """
        # Value loss (clipped)
        # 价值损失（裁剪）
        tdret = reward_sum.squeeze(-1) if reward_sum.dim() > 1 else reward_sum
        vp = value_pred.squeeze(-1) if value_pred.dim() > 1 else value_pred
        ov = old_value.squeeze(-1) if old_value.dim() > 1 else old_value

        vp_clip = ov + (vp - ov).clamp(-self.clip_param, self.clip_param)
        value_loss = (
            0.5
            * torch.maximum(
                (tdret - vp) ** 2,
                (tdret - vp_clip) ** 2,
            ).mean()
        )

        # Policy loss (PPO clip)
        # 策略损失（PPO clip）
        prob_dist = self._masked_softmax(logits, legal_action)
        entropy_loss = (-(prob_dist * torch.log(prob_dist.clamp(1e-9, 1))).sum(1)).mean()

        one_hot = torch.nn.functional.one_hot(old_action[:, 0].long(), self.label_size).float()
        new_prob = (one_hot * prob_dist).sum(1, keepdim=True)
        old_action_prob = (one_hot * old_prob).sum(1, keepdim=True)

        ratio = new_prob / old_action_prob.clamp(1e-9)

        adv = advantage.squeeze(-1) if advantage.dim() > 1 else advantage
        adv = adv.unsqueeze(-1)

        policy_loss = torch.maximum(
            -ratio * adv,
            -ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv,
        ).mean()

        # Total loss
        # 总损失
        total_loss = self.vf_coef * value_loss + policy_loss - self.var_beta * entropy_loss
        info = {
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }
        if rnd_pred is not None and rnd_target is not None:
            rnd_loss = torch.nn.functional.mse_loss(rnd_pred, rnd_target.detach())
            total_loss = total_loss + float(getattr(Config, "RND_LOSS_COEF", 0.0)) * rnd_loss
            info["rnd_loss"] = rnd_loss.item()

        return total_loss, info

    def _masked_softmax(self, logits, legal_action):
        """Apply legal action mask to logits before computing softmax.

        对 logits 应用合法动作掩码后计算 softmax。
        """
        label_max, _ = torch.max(logits * legal_action, dim=1, keepdim=True)
        logits = logits - label_max
        logits = logits * legal_action
        logits = logits + 1e5 * (legal_action - 1)
        return torch.nn.functional.softmax(logits, dim=1)

    @staticmethod
    def _linear_schedule(start, end, total_steps, step):
        """Linear interpolation from start to end over total_steps."""
        if total_steps <= 0:
            return float(end)
        ratio = min(max(float(step) / float(total_steps), 0.0), 1.0)
        return float(start + (end - start) * ratio)

    def _update_schedules(self):
        """Update optimizer learning rate and entropy coefficient by train_step."""
        self.current_lr = self._linear_schedule(
            self.lr_start,
            self.lr_end,
            self.lr_steps,
            self.train_step,
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.current_lr

        self.var_beta = self._linear_schedule(
            self.beta_start,
            self.beta_end,
            self.beta_steps,
            self.train_step,
        )
