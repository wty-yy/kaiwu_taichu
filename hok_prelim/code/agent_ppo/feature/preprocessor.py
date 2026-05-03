#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################

import math
from collections import deque

import numpy as np

from agent_ppo.conf.conf import Config

DIRECTION_ANGLES = {
    1: 0,
    2: 45,
    3: 90,
    4: 135,
    5: 180,
    6: 225,
    7: 270,
    8: 315,
}

# 8 方向动作对应位移（dx, dz），0-7 为移动方向，8-15 为同方向闪现
LEGAL_ACTION_IDX2POS = {
    0: (1, 0),    # 右
    1: (1, -1),   # 右上
    2: (0, -1),   # 上
    3: (-1, -1),  # 左上
    4: (-1, 0),   # 左
    5: (-1, 1),   # 左下
    6: (0, 1),    # 下
    7: (1, 1),    # 右下
}

# 地图 X 轴最大值（栅格坐标）
MAP_X_MAX = 128
# 地图 Z 轴最大值（栅格坐标）
MAP_Z_MAX = 128
# 物件类型：1=宝箱，2=加速 buff
SUB_TYPE_TREASURE = 1
SUB_TYPE_BUFF = 2
# 英雄与怪物的欧氏距离桶划分：0=[0,30), 1=[30,60), 2=[60,90), 3=[90,120), 4=[120,150), 5=[150,180]
L2_DISTANCE_SIZE = 6
APPROX_DISTANCE_BUCKETS = [0, 30, 60, 90, 120, 150]
# 怪物相对于英雄的方位：0=重叠/无效，1=东，2=东北，3=北，4=西北，5=西，6=西南，7=南，8=东南
RELATIVE_DIRECTION_SIZE = 9
# 最大怪物速度
MAX_MONSTER_SPEED = 2.0
# 最大怪物数
MAX_MONSTER_NUM = Config.MAX_MONSTER_NUM
# 最大宝箱数
MAX_TREASURES = Config.MAX_TREASURE_NUM
# 最大闪现冷却步数
MAX_FLASH_CD = 2000.0
# buff最大持续时间
MAX_BUFF_DURATION = 50.0
# buff最大数目
MAX_BUFF_NUM = Config.MAX_BUFF_NUM
MAP_WINDOW_SIZE = Config.MAP_FEATURE_SIZE
MAP_WINDOW_HALF = MAP_WINDOW_SIZE // 2



class FrameState:
    def __init__(self, frame_state):
        self.state = frame_state
        self.hero = FrameHero(frame_state['heroes'])
        self.monsters = [MonsterState(monster_state) for monster_state in frame_state['monsters']]
        self._update_monsters()  # 根据距离排序怪物列表
        self.treasures = [OrganState(state) for state in frame_state['organs'] if state['sub_type'] == SUB_TYPE_TREASURE]
        self.buffs = [OrganState(state) for state in frame_state['organs'] if state['sub_type'] == SUB_TYPE_BUFF]
        self.avail_sort_treasures = self._get_available_treasures()
        self.avail_sort_buffs = self._get_available_buffs()
    
    def _get_available_treasures(self):
        """Get list of available treasures that can be collected, sorted by distance (increasing)."""
        return sorted(
            [t for t in self.treasures if t.status == 1],
            key=lambda t: math.sqrt((t.pos['x'] - self.hero.pos['x'])**2 + (t.pos['z'] - self.hero.pos['z'])**2)
        )
    
    def _get_available_buffs(self):
        """Get list of available buffs that can be collected, sorted by distance (increasing)."""
        return sorted(
            [b for b in self.buffs if b.status == 1],
            key=lambda b: math.sqrt((b.pos['x'] - self.hero.pos['x'])**2 + (b.pos['z'] - self.hero.pos['z'])**2)
        )

    def _update_monsters(self):
        """Update monster list and re-sort by distance."""
        def monster_distance(monster: MonsterState):
            dis = math.inf
            if monster.is_in_view:
                dis = math.sqrt((monster.pos['x'] - self.hero.pos['x'])**2 + (monster.pos['z'] - self.hero.pos['z'])**2)
            return (monster.hero_l2_distance, dis)
        self.monsters = sorted(self.monsters, key=monster_distance)

class FrameHero:
    def __init__(self, hero_state):
        s = self.state = hero_state
        self.id = s['hero_id']                              # int32: 英雄实体 ID
        self.pos = s['pos']                                 # Position: 英雄位置 {x, z}（栅格坐标）
        self.treasure_score = s['treasure_score']           # float: 宝箱得分
        self.step_score = s['step_score']                   # float: 步数得分
        self.treasure_collected_count = s['treasure_collected_count']  # int32: 已收集宝箱数
        self.flash_cooldown = s['flash_cooldown']           # int32: 闪现冷却
        self.buff_remain = s['buff_remaining_time']         # int32: buff剩余时间

class MonsterState:
    def __init__(self, monster_state):
        s = self.state = monster_state
        self.id = s['monster_id']                           # int32: 怪物实体 ID
        self.pos = s['pos']                                 # Position: 怪物当前位置 {x, z}
        self.hero_l2_distance = s['hero_l2_distance']       # int32: 与英雄的欧氏距离桶编号（0-5）。128x128 地图均匀划分：0=[0,30), 1=[30,60), 2=[60,90), 3=[90,120), 4=[120,150), 5=[150,180]
        self.hero_relative_direction = s['hero_relative_direction']  # int32: 怪物相对于英雄的方位（0-8）：0=重叠/无效，1=东，2=东北，3=北，4=西北，5=西，6=西南，7=南，8=东南
        self.speed = s['speed']                             # int32: 怪物当前移动速度（格/步）
        self.monster_interval = s['monster_interval']       # int32: 第二怪物出现间隔配置值
        self.is_in_view = s['is_in_view']                   # int32: 是否在英雄视野内

class OrganState:  # Organ没有is_in_view字段，它们只在视野内才会出现在organs里
    def __init__(self, organ_state):
        s = self.state = organ_state
        self.id = s['config_id']                            # int32: 配置 ID（实体 ID）
        self.sub_type = s['sub_type']                       # int32: 物件类型（1=宝箱，2=加速 buff）
        self.pos = s['pos']                                 # Position: 物件位置 {x, z}
        self.status = s['status']                           # int32: 状态（1=可获取）
        self.hero_l2_distance = s['hero_l2_distance']       # int32: 与英雄的欧氏距离桶编号（0-5）
        self.hero_relative_direction = s['hero_relative_direction']  # int32: 物件相对于英雄的方位（0-8）

class EnvInfo:
    def __init__(self, env_info):
        info = self.info = env_info
        self.total_score = info['total_score']                    # float: 总得分
        self.step_no = info['step_no']                            # int32: 当前步数
        self.step_score = info['step_score']                      # float: 步数得分
        self.pos = info['pos']                                    # Position: 英雄位置 {x, z}
        self.treasure_score = info['treasure_score']              # float: 宝箱得分
        self.treasure_id = info['treasure_id']                    # int32[]: 剩余未收集宝箱 ID 列表
        self.monster_interval = info['monster_interval']          # int32: 第二怪物出现间隔
        self.max_step = info['max_step']                          # int32: 最大步数
        self.flash_count = info['flash_count']                    # int32: 闪现技能已使用次数
        self.total_buff = info['total_buff']                      # int32: buff 总数量配置
        self.collected_buff = info['collected_buff']              # int32: 已获取 buff 次数
        self.buff_refresh_time = info['buff_refresh_time']        # int32: buff 刷新时间配置
        self.total_treasure = info['total_treasure']              # int32: 总宝箱数量
        self.treasures_collected = info['treasures_collected']    # int32: 已收集宝箱数

class StateManager:
    """update -> get_obs -> get_reward"""

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.n_hit_wall = 0             # 本局撞墙次数
        self.last_action = -1           # 上一步执行动作
        self.frame_state = None         # 当前帧状态
        self.env_info = None            # 环境信息
        self.last_frame_state = None    # 上一步帧状态
        self.last_env_info = None       # 上一步环境信息
        # 以英雄为中心裁剪，需要为窗口半径预留边界
        self.map_pad = MAP_WINDOW_HALF
        self.repeat_count = np.zeros(
            (MAP_Z_MAX + 2 * self.map_pad, MAP_X_MAX + 2 * self.map_pad),
            dtype=np.float32,
        )
        self.last_reward_breakdown = self._new_reward_breakdown()
        # 全局地图，-2=任务地图外，-1=任务地图内未知区域，0=障碍物，1=可通行
        self.global_map = np.full(
            (MAP_Z_MAX + 2 * self.map_pad, MAP_X_MAX + 2 * self.map_pad),
            -2,
            dtype=np.int8,
        )
        self.global_map[
            self.map_pad:self.map_pad + MAP_Z_MAX,
            self.map_pad:self.map_pad + MAP_X_MAX,
        ] = -1
        self.new_opened_cells = 0
        self.known_treasures = {}
        self.known_buffs = {}
        self.hero_pos_history = deque(maxlen=Config.PROCRASTINATION_WINDOW_STEPS + 1)

    @staticmethod
    def _new_reward_breakdown():
        return {
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
        }

    def update(self, env_obs, last_action=-1):
        env_obs = env_obs['observation']
        self.last_frame_state = self.frame_state
        self.last_env_info = self.env_info
        self.last_action = int(last_action) if last_action is not None else -1
        if self.last_frame_state is not None:
            # 坐标索引为 [z, x]，写入 pad 后的地图计数
            self.repeat_count[
                self.last_frame_state.hero.pos['z'] + self.map_pad,
                self.last_frame_state.hero.pos['x'] + self.map_pad,
            ] += 1

        self.step_no = env_obs['step_no']                          # int32: 当前步数
        self.frame_state = FrameState(env_obs['frame_state'])      # FrameState: 帧状态数据
        self.env_info = EnvInfo(env_obs['env_info'])               # EnvInfo: 环境信息
        hero_x, hero_z = self._to_int_pos(self.frame_state.hero.pos)
        self.hero_pos_history.append((hero_x, hero_z))
        self.map_info = env_obs['map_info']                        # int32[][]: 局部地图信息（以英雄为中心的视野栅格，1=可通行，0=障碍物）
        self.legal_act = [int(x) for x in env_obs['legal_action']] # bool[16]: 直接使用环境原始合法动作掩码
        self._update_global_map()
        self._update_object_memory()
    
    def _update_global_map(self):
        """根据当前帧的局部地图和英雄位置更新全局地图，并统计本步新探索到的可通过格子数。"""
        map_info = np.asarray(self.map_info, dtype=np.int8)
        if map_info.shape != Config.MAP_INFO_SHAPE:
            raise ValueError(f"map_info shape mismatch, got {map_info.shape}, expect {Config.MAP_INFO_SHAPE}")
        h, w = Config.MAP_INFO_SHAPE
        half_h = h // 2
        half_w = w // 2
        hero_x, hero_z = int(self.frame_state.hero.pos['x']), int(self.frame_state.hero.pos['z'])
        top_left_x = hero_x - half_w
        top_left_z = hero_z - half_h

        gz0 = top_left_z + self.map_pad
        gx0 = top_left_x + self.map_pad
        target = self.global_map[gz0:gz0 + h, gx0:gx0 + w]

        x_coords = top_left_x + np.arange(w, dtype=np.int32)
        z_coords = top_left_z + np.arange(h, dtype=np.int32)
        valid_mask = (
            (z_coords[:, None] >= 0) &
            (z_coords[:, None] < MAP_Z_MAX) &
            (x_coords[None, :] >= 0) &
            (x_coords[None, :] < MAP_X_MAX)
        )
        self.new_opened_cells = int(
            np.count_nonzero((target == -1) & valid_mask & (map_info == 1))
        )
        target[valid_mask] = map_info[valid_mask]

    @staticmethod
    def _is_world_pos_valid(x, z):
        return 0 <= int(x) < MAP_X_MAX and 0 <= int(z) < MAP_Z_MAX

    @staticmethod
    def _to_int_pos(pos):
        return int(pos['x']), int(pos['z'])

    def _approx_pos_from_relative(self, anchor_pos, relative_direction, distance_bucket):
        anchor_x, anchor_z = self._to_int_pos(anchor_pos)
        if not (1 <= int(relative_direction) <= 8):
            return anchor_x, anchor_z

        dx, dz = LEGAL_ACTION_IDX2POS[int(relative_direction) - 1]
        norm = math.sqrt(dx * dx + dz * dz)
        approx_dis = APPROX_DISTANCE_BUCKETS[min(max(int(distance_bucket), 0), L2_DISTANCE_SIZE - 1)]

        if norm <= 1e-6 or approx_dis <= 0:
            return anchor_x, anchor_z

        ax = anchor_x + (dx / norm) * approx_dis
        az = anchor_z + (dz / norm) * approx_dis
        return (
            int(round(self.clip(ax, 0, MAP_X_MAX - 1))),
            int(round(self.clip(az, 0, MAP_Z_MAX - 1))),
        )

    def _get_centered_window(self, grid):
        hero_x, hero_z = self._to_int_pos(self.frame_state.hero.pos)
        center_x = hero_x + self.map_pad
        center_z = hero_z + self.map_pad
        x0 = center_x - MAP_WINDOW_HALF
        z0 = center_z - MAP_WINDOW_HALF
        return grid[z0:z0 + MAP_WINDOW_SIZE, x0:x0 + MAP_WINDOW_SIZE]

    def _mark_world_on_centered_channel(self, channel, world_x, world_z, value):
        hero_x, hero_z = self._to_int_pos(self.frame_state.hero.pos)
        local_x = int(world_x) - hero_x + MAP_WINDOW_HALF
        local_z = int(world_z) - hero_z + MAP_WINDOW_HALF
        if 0 <= local_z < MAP_WINDOW_SIZE and 0 <= local_x < MAP_WINDOW_SIZE:
            channel[local_z, local_x] = max(float(channel[local_z, local_x]), float(value))

    def _update_object_memory(self):
        remaining_treasure_ids = {int(tid) for tid in self.env_info.treasure_id}

        # 仅保留未获取宝箱
        for tid in list(self.known_treasures.keys()):
            if tid not in remaining_treasure_ids:
                del self.known_treasures[tid]

        for treasure in self.frame_state.treasures:
            tx, tz = self._to_int_pos(treasure.pos)
            if not self._is_world_pos_valid(tx, tz):
                continue
            if treasure.status == 1:
                self.known_treasures[int(treasure.id)] = (tx, tz)
            elif int(treasure.id) in self.known_treasures:
                del self.known_treasures[int(treasure.id)]

        # buff 无全量 ID 列表，结合可见状态和采集计数近似维护“未获取 buff”
        for buff in self.frame_state.buffs:
            bx, bz = self._to_int_pos(buff.pos)
            if not self._is_world_pos_valid(bx, bz):
                continue
            if buff.status == 1:
                self.known_buffs[int(buff.id)] = (bx, bz)
            elif int(buff.id) in self.known_buffs:
                del self.known_buffs[int(buff.id)]

        if self.last_env_info is not None:
            buff_gain = int(self.env_info.collected_buff - self.last_env_info.collected_buff)
            hero_x, hero_z = self._to_int_pos(self.frame_state.hero.pos)
            while buff_gain > 0 and self.known_buffs:
                nearest_key = min(
                    self.known_buffs,
                    key=lambda bid: (self.known_buffs[bid][0] - hero_x) ** 2 + (self.known_buffs[bid][1] - hero_z) ** 2,
                )
                min_dist_sq = (self.known_buffs[nearest_key][0] - hero_x) ** 2 + (self.known_buffs[nearest_key][1] - hero_z) ** 2
                if min_dist_sq <= 9.0: 
                    del self.known_buffs[nearest_key]
                buff_gain -= 1

    def _build_map_feature(self):
        global_window = self._get_centered_window(self.global_map)
        repeat_window = self._get_centered_window(self.repeat_count)

        ch_outside = (global_window == -2).astype(np.float32)
        ch_unexplored = (global_window == -1).astype(np.float32)
        ch_obstacle = (global_window == 0).astype(np.float32)
        ch_passable = (global_window == 1).astype(np.float32)

        ch_monster = np.zeros((MAP_WINDOW_SIZE, MAP_WINDOW_SIZE), dtype=np.float32)
        for monster in self.frame_state.monsters:
            if monster.is_in_view:
                mx, mz = self._to_int_pos(monster.pos)
                if self._is_world_pos_valid(mx, mz):
                    self._mark_world_on_centered_channel(ch_monster, mx, mz, 2.0)
            else:
                ax, az = self._approx_pos_from_relative(
                    self.frame_state.hero.pos,
                    monster.hero_relative_direction,
                    monster.hero_l2_distance,
                )
                self._mark_world_on_centered_channel(ch_monster, ax, az, 1.0)

        ch_treasure = np.zeros((MAP_WINDOW_SIZE, MAP_WINDOW_SIZE), dtype=np.float32)
        for tx, tz in self.known_treasures.values():
            self._mark_world_on_centered_channel(ch_treasure, tx, tz, 1.0)

        ch_buff = np.zeros((MAP_WINDOW_SIZE, MAP_WINDOW_SIZE), dtype=np.float32)
        for bx, bz in self.known_buffs.values():
            self._mark_world_on_centered_channel(ch_buff, bx, bz, 1.0)

        # 1. 计算地形 (Terrain)，取值 0, 1, 2, 3
        # 0:地图外, 1:未探索, 2:障碍物, 3:可通过
        terrain_type = np.zeros((MAP_WINDOW_SIZE, MAP_WINDOW_SIZE), dtype=np.uint8)
        terrain_type[ch_unexplored == 1] = 1
        terrain_type[ch_obstacle == 1] = 2
        terrain_type[ch_passable == 1] = 3
        # 剩下的默认为 0 (任务地图外的区域)

        # 2. 计算轨迹 (0, 1, 2) 分别对应原值的 0.0, 0.5, 1.0
        # 假设 repeat_window 里面存的是次数 0, 1, 2+
        visited_state = np.clip(repeat_window, 0, 2).astype(np.uint8)

        # 3. 计算怪物 (0, 1, 2) 分别对应原值的 0.0, 0.5, 1.0
        # 这需要你在原来赋值 0.5 的地方写 1，赋值 1.0 的地方写 2
        monster_state = ch_monster.astype(np.uint8) 

        # 4. 宝箱和 Buff (0, 1)
        treasure_state = ch_treasure.astype(np.uint8)
        buff_state = ch_buff.astype(np.uint8)

        # 位运算拼接
        # 地形占 bit 0-1, 轨迹占 bit 2-3, 怪物占 bit 4-5, 宝箱占 bit 6, Buff占 bit 7
        packed_map = (
            terrain_type 
            | (visited_state << 2) 
            | (monster_state << 4) 
            | (treasure_state << 6) 
            | (buff_state << 7)
        )
        
        # 最终返回展平的 MAP_WINDOW_SIZE x MAP_WINDOW_SIZE 的 float32 数据交给 Numpy 拼接
        # 大小从 8 * N * N 压缩到 N * N
        return packed_map.flatten().astype(np.float32)

    def _build_local_obstacle_feature(self):
        map_info = np.asarray(self.map_info, dtype=np.float32)
        if map_info.shape != Config.MAP_INFO_SHAPE:
            raise ValueError(f"map_info shape mismatch, got {map_info.shape}, expect {Config.MAP_INFO_SHAPE}")
        obstacle_feature = (map_info <= 0).astype(np.float32)
        return obstacle_feature.flatten()

    def get_obs(self):
        """Build obervation"""
        obs = []
        total_treasure = max(1, int(self.env_info.total_treasure))
        max_step = max(1, int(self.env_info.max_step))
        # Hero features
        obs.extend([
            # 英雄位置
            self.frame_state.hero.pos['x'] / MAP_X_MAX,
            self.frame_state.hero.pos['z'] / MAP_Z_MAX,
            # 剩余宝箱数占比
            (self.env_info.total_treasure - self.env_info.treasures_collected) / total_treasure,
            # 步数进度
            self.env_info.step_no / max_step,
            # 闪现冷却归一化
            self.frame_state.hero.flash_cooldown / MAX_FLASH_CD,
            # 加速 buff 剩余时间归一化
            self.frame_state.hero.buff_remain / MAX_BUFF_DURATION,
        ])
        # Monster features
        DIM_MONSTER = (
            # 怪物位置
            2 +
            # 怪物与英雄的距离桶 one-hot
            L2_DISTANCE_SIZE +
            # 怪物相对于英雄的方位 one-hot
            RELATIVE_DIRECTION_SIZE +
            # 怪物当前速度归一化
            1 +
            # 怪物是否在视野内
            1
        )
        x = [0.0] * (DIM_MONSTER * MAX_MONSTER_NUM)
        for i, monster in enumerate(self.frame_state.monsters[:MAX_MONSTER_NUM]):
            si = i * DIM_MONSTER
            if monster.is_in_view:
                x[si:si+2] = [monster.pos['x'] / MAP_X_MAX, monster.pos['z'] / MAP_Z_MAX]
            else:
                ax, az = self._approx_pos_from_relative(
                    self.frame_state.hero.pos,
                    monster.hero_relative_direction,
                    monster.hero_l2_distance,
                )
                x[si:si+2] = [self.clip(ax / MAP_X_MAX, 0.0, 1.0), self.clip(az / MAP_Z_MAX, 0.0, 1.0)]
            si += 2
            if 0 <= monster.hero_l2_distance < L2_DISTANCE_SIZE:
                x[si + monster.hero_l2_distance] = 1.0
            si += L2_DISTANCE_SIZE
            if 0 <= monster.hero_relative_direction < RELATIVE_DIRECTION_SIZE:
                x[si + monster.hero_relative_direction] = 1.0
            si += RELATIVE_DIRECTION_SIZE
            x[si] = monster.speed / MAX_MONSTER_SPEED
            si += 1
            x[si] = float(monster.is_in_view)
        obs.extend(x)
        # Treasure features
        DIM_TREASURE = (
            # 宝箱位置
            2 +
            # 宝箱与英雄的距离桶 one-hot
            L2_DISTANCE_SIZE +
            # 宝箱相对于英雄的方位 one-hot
            RELATIVE_DIRECTION_SIZE
        )
        x = [0.0] * (DIM_TREASURE * MAX_TREASURES)
        for i, treasure in enumerate(self.frame_state.avail_sort_treasures[:MAX_TREASURES]):
            si = i * DIM_TREASURE
            x[si:si+2] = [treasure.pos['x'] / MAP_X_MAX, treasure.pos['z'] / MAP_Z_MAX]
            si += 2
            if 0 <= treasure.hero_l2_distance < L2_DISTANCE_SIZE:
                x[si + treasure.hero_l2_distance] = 1.0
            si += L2_DISTANCE_SIZE
            if 0 <= treasure.hero_relative_direction < RELATIVE_DIRECTION_SIZE:
                x[si + treasure.hero_relative_direction] = 1.0
        obs.extend(x)
        # Buff features
        DIM_BUFF = (
            # Buff位置
            2 +
            # Buff与英雄的距离桶 one-hot
            L2_DISTANCE_SIZE +
            # Buff相对于英雄的方位 one-hot
            RELATIVE_DIRECTION_SIZE
        )
        x = [0.0] * (DIM_BUFF * MAX_BUFF_NUM)
        for i, buff in enumerate(self.frame_state.avail_sort_buffs[:MAX_BUFF_NUM]):
            si = i * DIM_BUFF
            x[si:si+2] = [buff.pos['x'] / MAP_X_MAX, buff.pos['z'] / MAP_Z_MAX]
            si += 2
            if 0 <= buff.hero_l2_distance < L2_DISTANCE_SIZE:
                x[si + buff.hero_l2_distance] = 1.0
            si += L2_DISTANCE_SIZE
            if 0 <= buff.hero_relative_direction < RELATIVE_DIRECTION_SIZE:
                x[si + buff.hero_relative_direction] = 1.0
        obs.extend(x)
        # 周围 21x21 障碍物特征（以英雄为中心，1=障碍，0=可通行）
        local_obstacle_feature = self._build_local_obstacle_feature()
        # 压缩地图特征（以英雄为中心）
        map_feature = self._build_map_feature()
        obs_arr = np.array(obs, dtype=np.float32)
        x_obs = np.concatenate((obs_arr, local_obstacle_feature, map_feature.flatten()), axis=0, dtype=np.float32)
        assert len(x_obs) == Config.FEATURE_LEN, f"obs len mismatch, got {len(x_obs)}, expect {Config.FEATURE_LEN}"
        return x_obs

    @staticmethod
    def _pos_l2(pos_a, pos_b):
        dx = float(pos_a['x'] - pos_b['x'])
        dz = float(pos_a['z'] - pos_b['z'])
        return math.sqrt(dx * dx + dz * dz)
    
    @staticmethod
    def clip(value, min_value, max_value):
        return max(min_value, min(max_value, value))

    def _is_flash_action(self):
        # 动作空间约定：0-7 为移动，8-15 为闪现方向
        return 8 <= self.last_action < Config.ACTION_NUM

    def _move_distance(self):
        if self.last_frame_state is None or self.frame_state is None:
            return 0.0
        return self._pos_l2(self.last_frame_state.hero.pos, self.frame_state.hero.pos)

    def _get_displacement_in_recent_steps(self, window_steps):
        if len(self.hero_pos_history) <= int(window_steps):
            return None

        start_x, start_z = self.hero_pos_history[-int(window_steps) - 1]
        end_x, end_z = self.hero_pos_history[-1]
        dx = float(end_x - start_x)
        dz = float(end_z - start_z)
        return math.sqrt(dx * dx + dz * dz)

    def _reset_procrastination_window(self):
        if self.frame_state is None:
            self.hero_pos_history.clear()
            return
        hero_x, hero_z = self._to_int_pos(self.frame_state.hero.pos)
        self.hero_pos_history.clear()
        self.hero_pos_history.append((hero_x, hero_z))

    def _get_visible_monster_distances(self, state):
        if state is None:
            return []
        return [
            self._pos_l2(monster.pos, state.hero.pos)
            for monster in state.monsters
            if monster.is_in_view
        ]

    @staticmethod
    def _iter_grid_line_points(x0, z0, x1, z1):
        """Bresenham 网格连线，返回从起点到终点（含端点）的整点路径。"""
        points = []
        dx = abs(x1 - x0)
        dz = abs(z1 - z0)
        sx = 1 if x0 < x1 else -1
        sz = 1 if z0 < z1 else -1
        err = dx - dz

        while True:
            points.append((x0, z0))
            if x0 == x1 and z0 == z1:
                break
            e2 = err * 2
            if e2 > -dz:
                err -= dz
                x0 += sx
            if e2 < dx:
                err += dx
                z0 += sz
        return points

    def _count_obstacles_on_path(self, start_pos, end_pos):
        """统计两点连线（不含起终点）上已知障碍物数量。"""
        x0, z0 = int(start_pos['x']), int(start_pos['z'])
        x1, z1 = int(end_pos['x']), int(end_pos['z'])
        line_points = self._iter_grid_line_points(x0, z0, x1, z1)
        if len(line_points) <= 2:
            return 0

        wall_count = 0
        for x, z in line_points[1:-1]:
            gx = x + self.map_pad
            gz = z + self.map_pad
            if not (0 <= gz < self.global_map.shape[0] and 0 <= gx < self.global_map.shape[1]):
                continue
            if self.global_map[gz, gx] == 0:
                wall_count += 1
        return wall_count

    def _has_visible_monster_in_flash_rect(self, start_pos, end_pos):
        x0, z0 = self._to_int_pos(start_pos)
        x1, z1 = self._to_int_pos(end_pos)
        x_min, x_max = min(x0, x1), max(x0, x1)
        z_min, z_max = min(z0, z1), max(z0, z1)

        for state in (self.last_frame_state, self.frame_state):
            if state is None:
                continue
            for monster in state.monsters:
                if not monster.is_in_view:
                    continue
                mx, mz = self._to_int_pos(monster.pos)
                if not self._is_world_pos_valid(mx, mz):
                    continue
                if x_min <= mx <= x_max and z_min <= mz <= z_max:
                    return True
        return False

    def _all_last_visible_monsters_farther(self, start_pos, end_pos):
        if self.last_frame_state is None:
            return True

        has_visible_monster = False
        for monster in self.last_frame_state.monsters:
            if not monster.is_in_view:
                continue
            mx, mz = self._to_int_pos(monster.pos)
            if not self._is_world_pos_valid(mx, mz):
                continue

            has_visible_monster = True
            dist_before = self._pos_l2(start_pos, monster.pos)
            dist_after = self._pos_l2(end_pos, monster.pos)
            if dist_after <= dist_before + 1e-6:
                return False

        return True if has_visible_monster else True

    def get_reward(self):
        breakdown = self._new_reward_breakdown()
        max_monster_speed = 1.0
        for monster in self.frame_state.monsters:
            max_monster_speed = max(max_monster_speed, monster.speed)  # 1 or 2
        # 存活奖励：随怪物速度增加而增加，鼓励在更难的环境中生存
        breakdown["reward_alive"] = float(Config.REW_ALIVE) * max_monster_speed * len(self.frame_state.monsters)
        # 探索奖励：每步新探索到的可通过格子数乘以奖励系数，鼓励探索地图
        breakdown["reward_explore_map"] = float(Config.REW_EXPLORE_MAP * self.new_opened_cells)

        if self.last_env_info is not None and self.last_frame_state is not None:
            # 宝箱奖励
            treasure_gain = self.env_info.treasures_collected - self.last_env_info.treasures_collected
            breakdown["reward_treasure"] = float(Config.REW_TREASURE * treasure_gain)

            # Buff奖励
            buff_gain = self.env_info.collected_buff - self.last_env_info.collected_buff
            breakdown["reward_buff"] = float(Config.REW_BUFF * buff_gain)

            if treasure_gain > 0 or buff_gain > 0:
                self._reset_procrastination_window()

            last_visible_monster_dists = self._get_visible_monster_distances(self.last_frame_state)
            current_visible_monster_dists = self._get_visible_monster_distances(self.frame_state)

            # 向最近宝箱靠近奖励：仅在附近怪物风险可接受时才给，避免被宝箱塑形推到怪物脸上
            approach_is_safe = True
            if current_visible_monster_dists:
                approach_is_safe = min(current_visible_monster_dists) >= float(Config.APPROACH_SAFE_MONSTER_DISTANCE)
            if (
                approach_is_safe and
                self.last_frame_state.avail_sort_treasures and self.frame_state.avail_sort_treasures and
                treasure_gain == 0  # 当收集了当前宝箱后，立刻计算下一个宝箱出现错误距离
            ):
                last_nearest = self.last_frame_state.avail_sort_treasures[0]
                last_dist = self._pos_l2(last_nearest.pos, self.last_frame_state.hero.pos)
                current_nearest = self.frame_state.avail_sort_treasures[0]
                current_dist = self._pos_l2(current_nearest.pos, self.frame_state.hero.pos)
                breakdown["reward_approach"] = float(Config.REW_APPROACH * (last_dist - current_dist))

            # 危险区脱险奖励：鼓励在怪物已经很近时主动拉开距离，避免原地抖动或小范围绕圈
            if last_visible_monster_dists and current_visible_monster_dists:
                last_min_dist = min(last_visible_monster_dists)
                current_min_dist = min(current_visible_monster_dists)
                if last_min_dist < 6.0 and current_min_dist > last_min_dist:
                    breakdown["reward_escape_danger"] = float(
                        Config.REW_ESCAPE_DANGER * min(current_min_dist - last_min_dist, 3.0)
                    )

            # 危险抢箱惩罚：拿到宝箱时如果怪物仍在近距离，视为冒进，避免“为了吃箱送脸”
            if treasure_gain > 0 and current_visible_monster_dists:
                current_min_dist = min(current_visible_monster_dists)
                if current_min_dist < 5.0:
                    breakdown["reward_treasure_danger"] = float(
                        Config.REW_TREASURE_DANGER * treasure_gain * (5.0 - current_min_dist)
                    )

            move_dist = self._move_distance()
            used_flash = self._is_flash_action()

            # 无效移动惩罚：未放技能且未发生位移
            if (not used_flash) and move_dist < 1e-6:
                breakdown["reward_invalid_move"] = float(Config.REW_INVALID_MOVE)
                self.n_hit_wall += 1

            # 闪现只有惩罚奖励：满足豁免条件则不惩罚，否则固定惩罚
            if used_flash:
                through_wall = False
                through_monster = False
                all_monsters_farther = False

                if move_dist > 6.0:
                    wall_count = self._count_obstacles_on_path(
                        self.last_frame_state.hero.pos,
                        self.frame_state.hero.pos,
                    )
                    through_wall = wall_count > 0
                    through_monster = self._has_visible_monster_in_flash_rect(
                        self.last_frame_state.hero.pos,
                        self.frame_state.hero.pos,
                    )
                    all_monsters_farther = self._all_last_visible_monsters_farther(
                        self.last_frame_state.hero.pos,
                        self.frame_state.hero.pos,
                    )

                no_penalty = move_dist > 6.0 and (through_wall or through_monster) and all_monsters_farther
                if not no_penalty:
                    breakdown["reward_bad_flash"] = float(Config.REW_BAD_FLASH)

            # 重复经过惩罚：当前位置周围 5x5 历史步数和超过阈值后线性惩罚
            cx = self.frame_state.hero.pos['x'] + self.map_pad
            cz = self.frame_state.hero.pos['z'] + self.map_pad
            sum_repeat_count = float(self.repeat_count[cz - 2:cz + 3, cx - 2:cx + 3].sum())
            breakdown["reward_repeat"] = float(
                max(0.0, sum_repeat_count - Config.REW_REPEAT_SUM_THRESHOLD) * Config.REW_REPEAT_SCALE
            )

            # 拖延惩罚：近 10 步位移小于 5 时惩罚
            procrastination_dist = self._get_displacement_in_recent_steps(
                Config.PROCRASTINATION_WINDOW_STEPS
            )
            if (
                procrastination_dist is not None and
                procrastination_dist < float(Config.PROCRASTINATION_MIN_MOVE_DISTANCE)
            ):
                breakdown["reward_procrastination"] = float(Config.REW_PROCRASTINATION)

        if Config.REW_GLOBAL_SCALE != 1.0:
            for key in breakdown:
                breakdown[key] *= Config.REW_GLOBAL_SCALE

        self.last_reward_breakdown = breakdown
        reward = float(sum(breakdown.values()))
        return [reward]

    def get_all(self):
        feature = self.get_obs()
        reward = self.get_reward()
        return feature, self.legal_act, reward


class Preprocessor:
    def __init__(self):
        self.state_manager = StateManager()

    def reset(self):
        self.state_manager.reset()

    def feature_process(self, env_obs, last_action=-1):
        """保持2026接口协议：返回 (feature, legal_action, reward)"""
        self.state_manager.update(env_obs, last_action)
        return self.state_manager.get_all()

    def get_step_reward_breakdown(self):
        return dict(self.state_manager.last_reward_breakdown)

    def get_n_hit_wall(self):
        return int(self.state_manager.n_hit_wall)
