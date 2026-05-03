#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################

import numpy as np


class WeightedMapSampler:
    """在每次 reset 前按权重选择地图，并写回用户配置。"""

    def __init__(self, usr_conf, map_sample_weights, default_weight=1.0, logger=None):
        self.usr_conf = usr_conf
        self.map_sample_weights = map_sample_weights or {}
        self.logger = logger

        try:
            self.default_weight = max(float(default_weight), 0.0)
        except (TypeError, ValueError):
            self.default_weight = 1.0

        self.env_conf = self._get_env_conf_container()
        self.map_random = bool(self._conf_get(self.env_conf, "map_random", True))

        self.map_ids = []
        self.map_probs = None
        self.sample_counts = {}
        if self.map_random:
            self.map_ids, self.map_probs = self._build_weighted_map_sampling()
            self.sample_counts = {int(map_id): 0 for map_id in self.map_ids}
            if self.map_ids and self.logger is not None:
                probs = [round(float(p), 4) for p in self.map_probs]
                self.logger.info(
                    f"weighted map sampling enabled, maps={self.map_ids}, probs={probs}"
                )
        elif self.logger is not None:
            self.logger.info("map_random is false, weighted map sampling disabled")

    @staticmethod
    def _conf_get(conf_container, key, default=None):
        if isinstance(conf_container, dict):
            return conf_container.get(key, default)
        return getattr(conf_container, key, default)

    @staticmethod
    def _conf_set(conf_container, key, value):
        if isinstance(conf_container, dict):
            conf_container[key] = value
        else:
            setattr(conf_container, key, value)

    def _get_env_conf_container(self):
        if isinstance(self.usr_conf, dict):
            env_conf = self.usr_conf.get("env_conf")
            if isinstance(env_conf, dict):
                return env_conf
            return self.usr_conf

        env_conf = getattr(self.usr_conf, "env_conf", None)
        return env_conf if env_conf is not None else self.usr_conf

    def _build_weighted_map_sampling(self):
        raw_maps = self._conf_get(self.env_conf, "map", [])
        if not isinstance(raw_maps, (list, tuple)) or len(raw_maps) == 0:
            if self.logger is not None:
                self.logger.warning("map config is empty, skip weighted map sampling")
            return [], None

        # map 配置要求不重复，这里去重一次并回写，避免误配置导致环境侧校验失败。
        uniq_maps = []
        seen = set()
        for map_id in raw_maps:
            try:
                map_id_int = int(map_id)
            except (TypeError, ValueError):
                continue
            if map_id_int in seen:
                continue
            seen.add(map_id_int)
            uniq_maps.append(map_id_int)

        if not uniq_maps:
            if self.logger is not None:
                self.logger.warning("no valid map id found, skip weighted map sampling")
            return [], None

        self._conf_set(self.env_conf, "map", list(uniq_maps))

        weights = []
        for map_id in uniq_maps:
            map_weight = self.map_sample_weights.get(map_id, self.default_weight)
            try:
                map_weight = float(map_weight)
            except (TypeError, ValueError):
                map_weight = self.default_weight
            weights.append(max(map_weight, 0.0))

        weight_sum = float(sum(weights))
        if weight_sum <= 0:
            probs = np.full(len(uniq_maps), 1.0 / len(uniq_maps), dtype=np.float64)
        else:
            probs = np.asarray(weights, dtype=np.float64) / weight_sum

        return uniq_maps, probs

    def prepare_for_reset(self):
        if not self.map_random or not self.map_ids:
            return

        selected_map = int(np.random.choice(self.map_ids, p=self.map_probs))
        self.sample_counts[selected_map] = self.sample_counts.get(selected_map, 0) + 1
        self._conf_set(self.env_conf, "map", [selected_map])

    def get_sample_counts(self):
        return dict(self.sample_counts)
