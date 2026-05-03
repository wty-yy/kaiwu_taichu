#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Data augmentation utilities for online rotated interaction.
清扫大作战在线旋转交互增强工具。
"""

from __future__ import annotations

import copy
from typing import Iterable, List

import numpy as np

from agent_ppo.conf.conf import Config


class RotationDataAugmentation:
    """Rotate observation/action between env space and rotated space."""

    def __init__(self, rotation_steps: Iterable[int], logger=None):
        self.logger = logger
        self.rotation_steps = self._normalize_rotation_steps(rotation_steps)

    @staticmethod
    def _normalize_rotation_steps(rotation_steps: Iterable[int]) -> List[int]:
        normalized = []
        for step in rotation_steps:
            s = int(step) % 4
            if s not in normalized:
                normalized.append(s)
        return normalized if normalized else [0, 1, 2, 3]

    def sample_rotation_step(self) -> int:
        return int(np.random.choice(self.rotation_steps))

    def rotate_obs_data(self, obs_data, step: int):
        """Rotate ObsData (feature + legal_action) into selected rotated space."""
        rotated_obs_data = copy.deepcopy(obs_data)
        feature = np.array(obs_data.feature, dtype=np.float32).reshape(-1)
        legal_action = np.array(obs_data.legal_action, dtype=np.float32).reshape(Config.ACTION_NUM)
        self._validate_obs_and_action(feature, legal_action)
        rotated_obs_data.feature = self.rotate_obs_feature(feature, step).astype(np.float32)
        rotated_obs_data.legal_action = self.rotate_action_vector(legal_action, step).astype(np.float32)
        return rotated_obs_data

    @staticmethod
    def _validate_obs_and_action(obs: np.ndarray, legal_action: np.ndarray):
        expected_obs_dim = int(Config.DIM_OF_OBSERVATION)
        if obs.size != expected_obs_dim:
            raise ValueError(f"obs dim mismatch: got={obs.size}, expected={expected_obs_dim}")

        expected_action_dim = int(Config.ACTION_NUM)
        if legal_action.size != expected_action_dim:
            raise ValueError(f"legal_action dim mismatch: got={legal_action.size}, expected={expected_action_dim}")

    def rotate_obs_feature(self, obs: np.ndarray, step: int) -> np.ndarray:
        if step == 0:
            return obs.copy()

        hero_dim = int(Config.HERO_BASE_FEATURE_DIM)
        npc_total_dim = int(Config.MAX_NPC_ROBOT * Config.NPC_ITEM_FEATURE_DIM)
        charger_total_dim = int(Config.MAX_CHARGER * Config.CHARGER_ITEM_FEATURE_DIM)
        map_shape = tuple(Config.MAP_FEATURE_SHAPE)
        map_total_dim = int(map_shape[0] * map_shape[1] * map_shape[2])

        hero = obs[:hero_dim].copy()
        npcs = obs[hero_dim : hero_dim + npc_total_dim].copy()
        chargers = obs[hero_dim + npc_total_dim : hero_dim + npc_total_dim + charger_total_dim].copy()
        map_flat = obs[-map_total_dim:].copy()

        hero[0], hero[1] = self._rotate_normalized_xy(hero[0], hero[1], step)
        npcs = self._rotate_entity_block(
            block=npcs,
            max_item=int(Config.MAX_NPC_ROBOT),
            item_dim=int(Config.NPC_ITEM_FEATURE_DIM),
            step=step,
        )
        chargers = self._rotate_entity_block(
            block=chargers,
            max_item=int(Config.MAX_CHARGER),
            item_dim=int(Config.CHARGER_ITEM_FEATURE_DIM),
            step=step,
        )

        map_tensor = map_flat.reshape(map_shape).copy()
        rotated_map = np.rot90(map_tensor, k=-step, axes=(1, 2)).reshape(-1)

        return np.concatenate([hero, npcs, chargers, rotated_map], dtype=np.float32)

    def _rotate_entity_block(self, block: np.ndarray, max_item: int, item_dim: int, step: int) -> np.ndarray:
        rotated = block.copy()
        for item_idx in range(max_item):
            start = item_idx * item_dim

            global_x = rotated[start]
            global_z = rotated[start + 1]
            rel_dx = rotated[start + 2]
            rel_dz = rotated[start + 3]

            rotated[start], rotated[start + 1] = self._rotate_normalized_xy(global_x, global_z, step)
            rotated[start + 2], rotated[start + 3] = self._rotate_relative_dx_dz(rel_dx, rel_dz, step)

            direction_onehot = rotated[start + 4 : start + 13]
            rotated[start + 4 : start + 13] = self._rotate_direction_onehot(direction_onehot, step)
        return rotated

    @staticmethod
    def _rotate_normalized_xy(x: float, z: float, step: int):
        x_new, z_new = float(x), float(z)
        for _ in range(step):
            x_new, z_new = 1.0 - z_new, x_new
        return x_new, z_new

    @staticmethod
    def _rotate_relative_dx_dz(dx: float, dz: float, step: int):
        dx_new, dz_new = float(dx), float(dz)
        for _ in range(step):
            dx_new, dz_new = -dz_new, dx_new
        return dx_new, dz_new

    def _rotate_direction_onehot(self, direction_onehot: np.ndarray, step: int) -> np.ndarray:
        rotated = np.zeros_like(direction_onehot)
        if direction_onehot.size != 9:
            return direction_onehot.copy()

        rotated[0] = direction_onehot[0]
        for old_dir_idx in range(1, 9):
            new_dir_idx = self.rotate_action_index(old_dir_idx - 1, step) + 1
            rotated[new_dir_idx] = direction_onehot[old_dir_idx]
        return rotated

    def rotate_action_vector(self, action_vec: np.ndarray, step: int) -> np.ndarray:
        rotated = np.zeros_like(action_vec)
        for old_idx, value in enumerate(action_vec):
            new_idx = self.rotate_action_index(old_idx, step)
            rotated[new_idx] = value
        return rotated

    @staticmethod
    def rotate_action_index(action_idx: int, step: int) -> int:
        new_idx = int(action_idx)
        for _ in range(step):
            new_idx = (new_idx - 2) % int(Config.ACTION_NUM)
        return new_idx

    @staticmethod
    def inverse_rotate_action_index(action_idx: int, step: int) -> int:
        new_idx = int(action_idx)
        for _ in range(step):
            new_idx = (new_idx + 2) % int(Config.ACTION_NUM)
        return new_idx
