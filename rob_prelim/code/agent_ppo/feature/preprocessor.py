#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor for Robot Vacuum.
清扫大作战特征预处理器。
"""

import math
from collections import deque

import numpy as np

from agent_ppo.conf.conf import Config


MAP_SIZE = 128
LOCAL_VIEW = 21
HALF_VIEW = LOCAL_VIEW // 2
MAX_NPC_ROBOT = 4
MAX_CHARGER = 4
# 方位 0=重叠/无效，1=东，2=东北，3=北，4=西北，5=西，6=西南，7=南，8=东南
MAX_RELATIVE_DIRECTION = 9
# 距离桶划分 0=[0,15), 1=[15,30), 2=[30,45), 3=[45,60), 4=[60,75), 5=[75,inf)
MAX_RELATIVE_DISTANCE_BIN = 6
# 充电桩动态奖励插值锚点：0,5,10,...,75格，共16点（地图128×128，3桩均匀分布，上限约75格）
_CHARGER_DIST_KNOTS = np.arange(0.0, 76.0, 5.0)
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

class Position:
    """Parse a position dictionary in the form {x, z}."""

    def __init__(self, pos):
        self.x = float(pos["x"])  # x: 位置横坐标
        self.z = float(pos["z"])  # z: 位置纵坐标
    
    @property
    def xz(self):
        """Return (x, z) tuple."""
        return self.x, self.z


class HeroState:
    """Parse hero state from observation payload."""

    def __init__(self, hero_state):
        self.hero_id = str(hero_state["hero_id"])     # hero_id: 小悟机器人实体 ID
        self.pos = Position(hero_state["pos"])        # pos: 小悟机器人当前位置
        self.battery = int(hero_state["battery"])     # battery: 当前电量
        self.battery_max = int(hero_state["battery_max"])  # battery_max: 电量上限
        self.score = int(hero_state["score"])         # score: 当前得分（清扫地面数量）
        self.dirt_cleaned = int(hero_state["dirt_cleaned"])  # dirt_cleaned: 已清扫污渍数量


class NpcState:
    """Parse official robot state from observation payload."""

    def __init__(self, npc_state):
        self.npc_id = str(npc_state["npc_id"])  # npc_id: 官方机器人实体 ID
        self.pos = Position(npc_state["pos"])   # pos: 官方机器人绝对坐标
        self.is_in_view = bool(npc_state["is_in_view"])   # 是否在视野内


class OrganState:
    """Parse organ state (charger) from observation payload."""

    def __init__(self, organ_state):
        self.sub_type = int(organ_state["sub_type"])    # sub_type: 物件类型（1=充电桩）
        self.config_id = int(organ_state["config_id"])  # config_id: 充电桩配置 ID
        self.pos = Position(organ_state["pos"])         # pos: 充电桩位置
        self.w = int(organ_state["w"])                  # w: 充电桩宽度（格子数）
        self.h = int(organ_state["h"])                  # h: 充电桩高度（格子数）


class FrameState:
    """Parse frame state payload."""

    def __init__(self, frame_state):
        self.hero = HeroState(frame_state['heroes'])  # heroes: 小悟机器人状态

        raw_npcs = frame_state["npcs"]  # npcs: 官方机器人状态列表
        self.npcs = [NpcState(item) for item in raw_npcs]
        self.npcs = [npc for npc in self.npcs if npc.pos.x != -1]

        raw_organs = frame_state["organs"]  # organs: 物件状态列表（含充电桩）
        self.organs = [OrganState(item) for item in raw_organs]
        self.organs = [organ for organ in self.organs if organ.pos.x != -1]


class EnvInfo:
    """Parse global environment info payload."""

    def __init__(self, env_info):
        self.total_score = int(env_info["total_score"])            # total_score: 总得分（清扫地面数量）
        self.step_no = int(env_info["step_no"])                    # step_no: 当前步数
        self.clean_score = int(env_info["clean_score"])            # clean_score: 清扫得分
        self.total_dirt = int(env_info["total_dirt"])              # total_dirt: 污渍总数量
        self.battery_max = int(env_info["battery_max"])            # battery_max: 电量上限
        self.remaining_charge = int(env_info["remaining_charge"])  # remaining_charge: 当前剩余电量
        self.charge_count = int(env_info["charge_count"])          # charge_count: 充电次数
        self.total_charger = int(env_info["total_charger"])        # total_charger: 充电桩数量
        self.npc_count = int(env_info["npc_count"])                # npc_count: 官方机器人数量
        self.pos = Position(env_info["pos"])                       # pos: 小悟机器人当前位置
        self.max_step = int(env_info["max_step"])                  # max_step: 最大步数
        self.step_cleaned_cells = [Position(cell) for cell in env_info["step_cleaned_cells"]]  # step_cleaned_cells: 本步清扫坐标列表


def get_relative_direction(dx, dz):
    """
    根据相对坐标 (dx, dz) 计算 1-8 的方向索引
    dx = 目标x - 英雄x
    dz = 目标z - 英雄z
    索引0为重合
    """
    # 重合
    if dx == 0 and dz == 0:
        return 0

    angle_rad = math.atan2(dz, dx)
    angle_deg = math.degrees(angle_rad)
    angle_deg = (angle_deg + 360) % 360
    # 计算 0-7 的离散索引 (加 22.5 实现四舍五入到最近的 45 度倍数)
    sector_idx = int((angle_deg + 22.5) // 45) % 8

    # 0->1(0°), 1->2(45°), 2->3(90°), 3->4(135°), 4->5(180°), 5->6(225°), 6->7(270°), 7->8(315°)
    return sector_idx + 1

def get_relative_distance_bin(dx, dz):
    """
    根据相对坐标 (dx, dz) 计算距离桶索引
    距离桶划分 0=[0,15), 1=[15,30), 2=[30,45), 3=[45,60), 4=[60,75), 5=[75,inf)
    """
    distance = math.sqrt(dx * dx + dz * dz)
    if distance < 15:
        return 0
    elif distance < 30:
        return 1
    elif distance < 45:
        return 2
    elif distance < 60:
        return 3
    elif distance < 75:
        return 4
    else:
        return 5

class StateManager:
    def __init__(self):
        """get_all -> update -> get_obs -> get_reward"""
        self.reset()

    def reset(self):
        """Reset per-episode states and short-term trackers."""
        self.step_no = 0
        self.last_action = None

        self.frame_state = None
        self.env_info = None
        self.last_frame_state = None
        self.last_env_info = None

        self.map_info = np.zeros((LOCAL_VIEW, LOCAL_VIEW), dtype=np.int32)
        self.legal_action = [0.0] * int(Config.ACTION_NUM)
        self.repeat_count = np.pad(     # 地图重复经过计数，扩展边界，出界部分填-1
            np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32),
            ((HALF_VIEW, HALF_VIEW), (HALF_VIEW, HALF_VIEW)),
            "constant",
            constant_values=-1,
        )
        self.map_last_step = np.full((MAP_SIZE, MAP_SIZE), -10, dtype=np.int32)  # 最近一次到达map位置的step

        hist_maxlen = int(Config.CLEAN_HISTORY_BUFFER_MAXLEN)
        self._clean_history_buffer = deque(maxlen=hist_maxlen)      # 每步清扫污渍格数历史
        self._no_clean_history_buffer = deque(maxlen=hist_maxlen)    # 每步是否未清扫（1=未清扫，0=有清扫）

        self.last_reward_breakdown = self._new_reward_breakdown()

    def update(self, env_obs, last_action):
        """Align state transition order with reference preprocessor update flow."""
        self.last_frame_state = self.frame_state
        self.last_env_info = self.env_info
        self.last_action = last_action
        if self.last_frame_state is not None:
            # 更新地图重复经过计数
            hero_pos = self.last_frame_state.hero.pos
            z_idx = int(hero_pos.z) + HALF_VIEW
            x_idx = int(hero_pos.x) + HALF_VIEW
            self.repeat_count[z_idx, x_idx] += 1
            self.map_last_step[int(hero_pos.z), int(hero_pos.x)] = self.last_env_info.step_no

        obs_dict = env_obs["observation"]  # observation: 顶层观测容器

        self.step_no = obs_dict["step_no"]                      # step_no: 当前步数
        self.frame_state = FrameState(obs_dict["frame_state"])  # frame_state: 帧状态数据
        self.env_info = EnvInfo(obs_dict["env_info"])           # env_info: 环境全局信息
        self.map_info = obs_dict["map_info"]                    # map_info: 地图信息（21×21 视野网格，0=障碍物，1=已清扫，2=污渍）
        self.legal_action = obs_dict["legal_action"]            # legal_act: 合法动作列表

    def _get_obs(self):
        """Build fixed-length observation vector"""
        obs = []

        # Our robot
        nearest_charger_dist = self._nearest_charger_distance(self.frame_state)
        nearest_charger_dist_norm = (
            min(float(nearest_charger_dist), float(MAP_SIZE)) / float(MAP_SIZE)
            if np.isfinite(nearest_charger_dist) else 1.0
        )
        obs.extend([
            self.frame_state.hero.pos.xz[0] / MAP_SIZE,  # 位置 x 坐标归一化
            self.frame_state.hero.pos.xz[1] / MAP_SIZE,  # 位置 z 坐标归一化
            # 当前电量
            self.frame_state.hero.battery / self.frame_state.hero.battery_max,
            # 清理污渍比例
            self.frame_state.hero.dirt_cleaned / (self.env_info.total_dirt + 1e-9),
            # 已完成步数比例
            self.env_info.step_no / self.env_info.max_step,
            # 充电次数连续归一化（上限5次=1.0，替代原二值首次充电信号）
            min(float(self.env_info.charge_count) / 5.0, 1.0),
            # 最近充电桩距离归一化（连续特征，inf→1.0）
            nearest_charger_dist_norm,
        ])

        # NPC robot
        x = [0.0] * ((
            2 +  # 全局坐标
            2 +  # 相对坐标
            MAX_RELATIVE_DIRECTION +  # 相对方向索引onehot
            MAX_RELATIVE_DISTANCE_BIN  # 相对距离桶索引onehot
        ) * MAX_NPC_ROBOT)
        start = 0
        sorted_npcs = sorted(
            self.frame_state.npcs,
            key=lambda npc: self._pos_l2(self.frame_state.hero.pos, npc.pos),
        )
        for npc in sorted_npcs[:MAX_NPC_ROBOT]:
            x[start] = npc.pos.xz[0] / MAP_SIZE
            x[start + 1] = npc.pos.xz[1] / MAP_SIZE
            dx, dz = self.get_hero_relative_distance(npc.pos.xz[0], npc.pos.xz[1])
            x[start + 2] = dx / MAP_SIZE
            x[start + 3] = dz / MAP_SIZE
            start += 4
            direction_idx = get_relative_direction(dx, dz)
            distance_bin_idx = get_relative_distance_bin(dx, dz)
            x[start + direction_idx] = 1.0  # 相对方向 onehot
            start += MAX_RELATIVE_DIRECTION
            x[start + distance_bin_idx] = 1.0  # 相对距离桶 onehot
            start += MAX_RELATIVE_DISTANCE_BIN
        obs.extend(x)

        # Charger
        x = [0.0] * ((
            2 +  # 全局坐标
            2 +  # 相对坐标
            MAX_RELATIVE_DIRECTION +  # 相对方向索引onehot
            MAX_RELATIVE_DISTANCE_BIN  # 相对距离桶索引onehot
        ) * MAX_CHARGER)
        start = 0
        sorted_chargers = sorted(
            [organ for organ in self.frame_state.organs if organ.sub_type == 1],
            key=lambda organ: self._pos_l2(self.frame_state.hero.pos, organ.pos),
        )
        for organ in sorted_chargers[:MAX_CHARGER]:
            x[start] = organ.pos.xz[0] / MAP_SIZE
            x[start + 1] = organ.pos.xz[1] / MAP_SIZE
            dx, dz = self.get_hero_relative_distance(organ.pos.xz[0], organ.pos.xz[1])
            x[start + 2] = dx / MAP_SIZE
            x[start + 3] = dz / MAP_SIZE
            start += 4
            direction_idx = get_relative_direction(dx, dz)
            distance_bin_idx = get_relative_distance_bin(dx, dz)
            x[start + direction_idx] = 1.0  # 相对方向 onehot
            start += MAX_RELATIVE_DIRECTION
            x[start + distance_bin_idx] = 1.0  # 相对距离桶 onehot
            start += MAX_RELATIVE_DISTANCE_BIN
        obs.extend(x)

        map_arr = np.array(self.map_info).flatten()
        # Map obstacle
        x_obstacle = (map_arr == 0).astype(np.float32)
        # Map cleaned
        x_cleaned = (map_arr == 1).astype(np.float32)
        # Map dirt
        x_dirt = (map_arr == 2).astype(np.float32)

        # Map repeat_count（第四通道）：截取视野窗口，最多统计到5次并归一化到[0,1]
        hero_z_idx = int(self.frame_state.hero.pos.z)
        hero_x_idx = int(self.frame_state.hero.pos.x)
        x_repeat_count = self.repeat_count[
            hero_z_idx : hero_z_idx + LOCAL_VIEW,
            hero_x_idx : hero_x_idx + LOCAL_VIEW,
        ]
        x_repeat_count = np.clip(x_repeat_count / 3, -1.0, 1.0)
        x_repeat_count = x_repeat_count.astype(np.float32).flatten()

        x_map = np.concatenate([x_obstacle, x_cleaned, x_dirt, x_repeat_count])

        ret = np.concatenate([obs, x_map], dtype=np.float32)

        return ret
    
    def _get_legal_action(self):
        """Return legal action mask for current state."""
        for idx, pos in LEGAL_ACTION_IDX2POS.items():
            if sum(self.legal_action) <= 1:
                break  # 如果合法动作数不超过1个，无需进一步筛选
            real_z = self.frame_state.hero.pos.z + pos[1]
            real_x = self.frame_state.hero.pos.x + pos[0]
            target_z = HALF_VIEW + pos[1]
            target_x = HALF_VIEW + pos[0]

            # 障碍物格子 不可行
            if self.map_info[int(target_z)][int(target_x)] == 0:
                self.legal_action[idx] = 0.0
            
            # 与npc距离小于一步的距离 不可行
            near_npc = False
            for npc in self.frame_state.npcs:
                if max(abs(npc.pos.z - real_z), abs(npc.pos.x - real_x)) <= 4:
                    near_npc = True
                    break
            if near_npc:
                self.legal_action[idx] = 0.0

        return self.legal_action

    def _get_reward(self):
        """Compute scalar reward from the latest state transition."""
        breakdown = self._new_reward_breakdown()
        current_charger_dist = self._nearest_charger_distance(self.frame_state)
        dist_clamped = float(np.clip(
            current_charger_dist if np.isfinite(current_charger_dist) else 75.0,
            0.0, 75.0,
        ))
        dynamic_low_battery_threshold = float(np.interp(
            dist_clamped, _CHARGER_DIST_KNOTS, Config.LOW_BATTERY_THRESHOLD_INTERP
        ))
        dynamic_edge_clean_coef = float(np.interp(
            dist_clamped, _CHARGER_DIST_KNOTS, Config.REW_EDGE_CLEAN_INTERP
        ))

        # 每步固定惩罚
        breakdown["step_punish"] = float(Config.REW_STEP_PUNISH)

        if self.last_frame_state is None or self.last_env_info is None:
            # 首帧仅保留步惩罚，并做全局缩放
            for key in breakdown:
                breakdown[key] *= float(Config.REW_GLOBAL_SCALE)
            self.last_reward_breakdown = breakdown
            return float(sum(breakdown.values()))

        # 有效清洁奖励（按本步清洁格数累计）
        cleaned_cells = self.frame_state.hero.dirt_cleaned - self.last_frame_state.hero.dirt_cleaned
        breakdown["valid_clean"] = float(Config.REW_VALID_CLEAN) * float(cleaned_cells)

        # 边缘清扫奖励（当前清扫格紧贴障碍或已清扫格时给予该分量）
        is_edge_clean = False
        for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 四邻域
            real_xi = self.frame_state.hero.pos.x + i
            real_zj = self.frame_state.hero.pos.z + j
            xi = HALF_VIEW + i
            zj = HALF_VIEW + j
            if 0 <= real_xi < MAP_SIZE and 0 <= real_zj < MAP_SIZE:
                if self.env_info.step_no - self.map_last_step[int(real_zj), int(real_xi)] <= 2:
                    continue  # 避免考虑前两步的清扫格子，防止斜上+下刷分
            if self.map_info[zj][xi] in [0, 1]:  # 障碍物或已清扫
                is_edge_clean = True
                break
        if is_edge_clean and cleaned_cells > 0:
            breakdown["edge_clean"] = float(dynamic_edge_clean_coef) * float(cleaned_cells)

        # 连续清扫奖励 / 连续未清扫惩罚：每步写入独立 buffer；满窗后若窗内记录「全为清扫」或「全为未清扫」则分别给奖/惩
        cleaned_cells_int = int(cleaned_cells)
        self._clean_history_buffer.append(cleaned_cells_int)
        self._no_clean_history_buffer.append(1 if cleaned_cells_int == 0 else 0)

        hist_win = self._clean_history_buffer.maxlen
        if hist_win is not None and len(self._clean_history_buffer) == hist_win:
            if all(int(v) > 0 for v in self._clean_history_buffer):
                breakdown["continuous_clean"] = float(Config.REW_CONTINUOUS_CLEAN)
            if all(int(v) == 1 for v in self._no_clean_history_buffer):
                breakdown["continuous_no_clean"] = float(Config.REW_CONTINUOUS_NO_CLEAN)

        # 不移动惩罚
        cur_pos = self.frame_state.hero.pos
        last_pos = self.last_frame_state.hero.pos
        if cur_pos.x == last_pos.x and cur_pos.z == last_pos.z:
            breakdown["no_move"] = float(Config.REW_NO_MOVE)

        # 危险距离惩罚: -max(5 - 最近NPC距离, 0) * 0.1
        current_npc_dist = self._nearest_npc_distance(self.frame_state)
        danger_gap = 0.0
        if np.isfinite(current_npc_dist):
            danger_gap = max(float(Config.REW_DANGER_RADIUS) - current_npc_dist, 0.0)
        breakdown["danger_distance"] = -danger_gap * float(Config.REW_DANGER_FACTOR)

        # 电量低于阈值时，靠近充电桩奖励
        battery_ratio = self.frame_state.hero.battery / max(float(self.frame_state.hero.battery_max), 1.0)
        if (
            battery_ratio < float(dynamic_low_battery_threshold) or  # 动态低电量阈值
            self.env_info.charge_count == 0  # 第一次充电前一直给予
        ):
            last_charger_dist = self._nearest_charger_distance(self.last_frame_state)
            if np.isfinite(last_charger_dist) and np.isfinite(current_charger_dist):
                breakdown["approach_charger_low_battery"] = (
                    (last_charger_dist - current_charger_dist) * float(Config.REW_LOW_BATTERY_APPROACH_CHARGER)
                )

        # 全局奖励缩放
        for key in breakdown:
            breakdown[key] *= float(Config.REW_GLOBAL_SCALE)

        self.last_reward_breakdown = breakdown
        return float(sum(breakdown.values()))

    def get_all(self):
        """Return feature, legal action mask and reward for current state."""
        obs = self._get_obs()
        legal_action = self._get_legal_action()
        rew = self._get_reward()
        return obs, legal_action, rew

    @staticmethod
    def _new_reward_breakdown():
        """Create reward component dictionary with zero initialization."""
        return {
            "valid_clean": 0.0,
            "edge_clean": 0.0,
            "continuous_clean": 0.0,
            "continuous_no_clean": 0.0,
            "step_punish": 0.0,
            "no_move": 0.0,
            "danger_distance": 0.0,
            "approach_charger_low_battery": 0.0,
        }

    def _nearest_npc_distance(self, frame_state: FrameState):
        """Get nearest NPC L2 distance to hero in a frame."""
        if len(frame_state.npcs) <= 0:
            return float("inf")
        hero_pos = frame_state.hero.pos
        return min(self._pos_l2(hero_pos, npc.pos) for npc in frame_state.npcs)

    def _nearest_charger_distance(self, frame_state: FrameState):
        """Get nearest charger L2 distance to hero in a frame."""
        chargers = [organ for organ in frame_state.organs if organ.sub_type == 1]
        if len(chargers) <= 0:
            return float("inf")
        hero_pos = frame_state.hero.pos
        return min(self._pos_l2(hero_pos, charger.pos) for charger in chargers)

    @staticmethod
    def _pos_l2(p1: Position, p2: Position):
        """Compute Euclidean distance between two Position objects."""
        return float(np.hypot(float(p1.x) - float(p2.x), float(p1.z) - float(p2.z)))

    def get_hero_relative_distance(self, x: float, z: float):
        """Calculate relative distance from hero to (x, z) coordinate."""
        dx = x - self.frame_state.hero.pos.x
        dz = z - self.frame_state.hero.pos.z
        return dx, dz

class Preprocessor:
    """Feature preprocessor for Robot Vacuum.

    清扫大作战特征预处理器。
    """

    def __init__(self):
        self.state_manager = StateManager()

    def reset(self):
        """Reset all internal state at episode start.

        对局开始时重置所有状态。
        """
        self.state_manager.reset()

    def feature_process(self, env_obs, last_action):
        """Generate feature vector, legal action mask, and scalar reward.

        生成特征向量、合法动作掩码和标量奖励。
        """
        self.state_manager.update(env_obs, last_action)
        feature, legal_action, reward = self.state_manager.get_all()
        return feature, legal_action, reward
