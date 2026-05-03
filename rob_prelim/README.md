# 2026开悟四足机器人初赛

## 赛题介绍

- [赛题说明](./assets/rob_prelim_introduction.png)
- [自测可视化结果](./assets/v2.5_183k自测可视化结果.png)
- [自测866得分](./assets/v2.5自测866表格.png)
- [最终分数榜](./assets/终榜得分排名.png)
- [晋级名单](./assets/2026具身初赛-四足机器人强化学习挑战-西部区域决赛晋级名单.jpg)

## 概述

使用 agent_ppo 训练，并行4环境，[训练日志logs/](./logs/)，主要修改文件 obs, model, reward 相关代码：[preprocessor.py](./code/agent_ppo/feature/preprocessor.py), [model.py](./code/agent_ppo/model/model.py)

## Observation

当前 obs 由 StateManager._get_obs 构建。
legal_action 作为动作掩码单独返回，不计入 obs 向量维度。

| 分段 | 组成 | 维度 |
| --- | --- | --- |
| 英雄基础特征 | 位置归一化 + 电量比例 + 清扫比例 + 步数进度 | 5 |
| 官方机器人特征 | 每个 NPC 19 维，最多 4 个 | 76 |
| 充电桩特征 | 每个充电桩 19 维，最多 4 个 | 76 |
| 地图四通道特征 | obstacle/cleaned/dirt/repeat_count 四个 21x21 通道展平拼接 | 1764 |
| 合计 | 5 + 76 + 76 + 1764 | 1921 |

| 分段 | 子项 | 维度 | 说明 |
| --- | --- | --- | --- |
| 英雄基础特征 | 位置 (x, z) | 2 | x / 128, z / 128 |
| 英雄基础特征 | 电量比例 | 1 | battery / battery_max |
| 英雄基础特征 | 清扫比例 | 1 | dirt_cleaned / (total_dirt + 1e-9) |
| 英雄基础特征 | 步数进度 | 1 | step_no / max_step |
| 官方机器人特征（单个） | 全局坐标 | 2 | npc.x / 128, npc.z / 128 |
| 官方机器人特征（单个） | 相对坐标 | 2 | (npc.x - hero.x)/128, (npc.z - hero.z)/128 |
| 官方机器人特征（单个） | 相对方向 one-hot | 9 | 索引 0 表示重合，1-8 为 8 邻域方向 |
| 官方机器人特征（单个） | 相对距离桶 one-hot | 6 | [0,30), [30,60), [60,90), [90,120), [120,150), [150,+inf) |
| 充电桩特征（单个） | 全局坐标 | 2 | charger.x / 128, charger.z / 128 |
| 充电桩特征（单个） | 相对坐标 | 2 | (charger.x - hero.x)/128, (charger.z - hero.z)/128 |
| 充电桩特征（单个） | 相对方向 one-hot | 9 | 与 NPC 同规则 |
| 充电桩特征（单个） | 相对距离桶 one-hot | 6 | 与 NPC 同规则 |
| 地图特征 | obstacle 通道 | 441 | map==0 置 1，否则 0 |
| 地图特征 | cleaned 通道 | 441 | map==1 置 1，否则 0 |
| 地图特征 | dirt 通道 | 441 | map==2 置 1，否则 0 |
| 地图特征 | repeat_count 通道 | 441 | 从 repeat_count 截取 21x21，clip 到 [0,1] |

补充说明：

- 相对方向通过 atan2 角度离散到 8 个 45 度扇区，重合点为索引 0。
- NPC 特征在写入槽位前，按与小悟的欧氏距离从近到远排序，最近实体优先占用前槽位。
- 充电桩特征在写入槽位前，先筛选 sub_type==1，再按与小悟的欧氏距离从近到远排序，最近实体优先占用前槽位。
- NPC/充电桩特征均按固定最大槽位展开（4 个），不足部分为 0。
- 最终 obs 为 float32 向量，长度固定 1921。
- update 中从 observation 读取 step_no/frame_state/env_info/map_info/legal_action，并缓存上一帧状态用于奖励计算。

## Reward

当前 reward 由 StateManager._get_reward 计算，使用 reward_breakdown 分量累加后返回。

| 版本号 | 类型 | 项目 | 计算方式 | 系数 |
| --- | --- | --- | --- | --- |
| v2 | 每步惩罚 | step_punish | 固定值 | REW_STEP_PUNISH |
| v2 | 正向奖励 | valid_clean | (curr.dirt_cleaned - last.dirt_cleaned) * REW_VALID_CLEAN | REW_VALID_CLEAN |
| v2.2 | 正向奖励 | edge_clean | 当本步有清扫且四邻域存在障碍物或已清扫格（排除上一步位置）时，cleaned_cells * REW_EDGE_CLEAN | REW_EDGE_CLEAN |
| v2.1 | 连续行为 | continuous_clean | 清扫历史 buffer 满窗且窗内每步清扫格数均 > 0 时，本步赋固定分量 | REW_CONTINUOUS_CLEAN |
| v2.1 | 连续行为 | continuous_no_clean | 未清扫标记 buffer 满窗且窗内每步均未清扫时，本步赋固定分量（配置为负即惩罚） | REW_CONTINUOUS_NO_CLEAN |
| v2 | 行为约束 | no_move | 若当前位置与上一步相同则加 REW_NO_MOVE | REW_NO_MOVE |
| v2 | 风险惩罚 | danger_distance | -max(REW_DANGER_RADIUS - 最近NPC距离, 0) * REW_DANGER_FACTOR | REW_DANGER_RADIUS, REW_DANGER_FACTOR |
| v2 | 引导奖励 | approach_charger_low_battery | 低电量时 (last_最近充电桩距离 - curr_最近充电桩距离) * REW_LOW_BATTERY_APPROACH_CHARGER | LOW_BATTERY_THRESHOLD, REW_LOW_BATTERY_APPROACH_CHARGER |

reward 计算流程：

1. 初始化 breakdown，先写入 step_punish。
2. 若没有上一帧（首帧）：仅对现有分量乘 REW_GLOBAL_SCALE 后返回。
3. 计算 valid_clean（来自 hero.dirt_cleaned 的前后差值）。
4. 若本步有清扫且当前清扫格邻近障碍物/已清扫格，叠加 edge_clean。
5. 写入连续清扫与连续未清扫 buffer；满窗后按规则叠加 continuous_clean / continuous_no_clean。
6. 若位置不变，叠加 no_move。
7. 计算最近 NPC 欧氏距离，叠加 danger_distance。
8. 若 battery / battery_max < LOW_BATTERY_THRESHOLD，计算靠近充电桩分量。
9. 对所有分量统一乘 REW_GLOBAL_SCALE。
10. 返回 sum(breakdown.values())。

| breakdown 键 | 含义 |
| --- | --- |
| valid_clean | 有效清洁奖励分量 |
| edge_clean | 靠边清扫奖励分量 |
| continuous_clean | 连续清扫奖励分量 |
| continuous_no_clean | 连续未清扫惩罚分量 |
| step_punish | 步数惩罚分量 |
| no_move | 不移动惩罚分量 |
| danger_distance | 危险距离惩罚分量 |
| approach_charger_low_battery | 低电量靠近充电桩分量 |

补充说明：

- 最近距离均使用欧氏距离 L2。
- 当场上不存在 NPC/充电桩时，对应最近距离记为 +inf。
- 终局奖励（例如撞 NPC 结束、电量耗尽结束）不在 preprocessor 内计算，由 workflow 在 done 分支单独处理。

## Model

当前模型采用“Set Encoder + Map CNN + 融合 MLP + Actor/Critic 双头”结构，
实现位于 `agent_ppo/model/model.py`，核心模块如下。

| 模块 | 输入 | 核心结构 | 输出 |
| --- | --- | --- | --- |
| Hero Encoder | 5 维（英雄基础特征） | Linear 5->32, ELU, Linear 32->64 | 64 |
| NPC Set Encoder | 4 个槽位，每槽 19 维 | 共享 MLP + Masked Max Pooling | 64 |
| Charger Set Encoder | 4 个槽位，每槽 19 维 | 共享 MLP + Masked Max Pooling | 64 |
| Map CNN Encoder | 4x21x21（obstacle/cleaned/dirt/repeat_count） | Conv(4->16)->Conv(16->32,s=2)->Conv(32->64,s=2)->MLP | 128 |
| Fusion MLP | 拼接后 320 维 | Linear 320->256, ELU, Linear 256->128, ELU | 128 |
| Actor Head | 128 | Linear 128->ACTION_NUM | 8 |
| Critic Head | 128 | Linear 128->VALUE_NUM | 1 |

| 观测切分 | 维度 |
| --- | --- |
| Hero 基础特征 | 5 |
| NPC 特征总维度（4x19） | 76 |
| Charger 特征总维度（4x19） | 76 |
| Map 特征总维度（4x21x21） | 1764 |
| 总计 | 1921 |

补充说明：

- Set Encoder 支持两种输入形状：`[batch, max_num * item_dim]` 或 `[batch, max_num, item_dim]`。
- 空槽位通过 mask 屏蔽，池化时置为 `-1e9`；若全空集合，回落为全 0 向量。
- `forward` 会校验输入维度与 `feature_len` 一致，不一致会抛出异常。
- 线性层与卷积层统一使用正交初始化，bias 初始化为 0。Actor head 权重乘以 0.01 以稳定初始训练。
- 当前实现优先读取 `Config` 中的维度配置；若缺失则使用默认切分（5/76/76/1764）。