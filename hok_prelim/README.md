# 2026开悟王者初赛

## 赛题介绍

- [赛题说明](./assets/hok_prelim_introduction.png)
- [地图可分可视化1](./assets/王者128得分地图1.png)
- [地图可分可视化2](./assets/王者1494得分地图1.png)
- [自测评估](./assets/王者v2.2最终自测评估.png)
- [最终得分榜](./assets/hok_prelim_final_rank.png)
- [晋级名单](./assets/D01-智能体决策算法-西部区域决赛晋级名单.jpg)

## 概述

使用 agent_ppo 训练，并行24环境，[训练日志logs/](./logs/)，主要修改文件 obs, model, reward 相关代码：[preprocessor.py](./code/agent_ppo/feature/preprocessor.py), [model.py](./code/agent_ppo/model/model.py)

## Obs

当前 obs 由 `StateManager.get_obs` 构建，`legal_action` 直接使用环境返回的原始动作掩码，单独输入策略采样流程，不计入 obs 维度。

| 分段 | 组成 | 维度 |
| --- | --- | --- |
| 英雄基础特征 | 位置归一化 + 剩余宝箱占比 + 步数进度 + 闪现冷却归一化 + buff 持续时间归一化 | 6 |
| 怪物特征 | 每个怪物 19 维，最多 2 个 | 38 |
| 宝箱特征 | 每个宝箱 17 维，最多 10 个 | 170 |
| buff 特征 | 每个 buff 17 维，最多 2 个 | 34 |
| 局部障碍物特征 | `21 x 21` 视野障碍平面，`1=障碍 / 0=可通行` | 441 |
| 地图特征 | 1 x 41 x 41 位掩码压缩地图（以智能体为中心） | 1681 |
| 合计 | 6 + 38 + 170 + 34 + 441 + 1681 | 2370 |

| 分段 | 子项 | 维度 |
| --- | --- | --- |
| 英雄基础特征 | 英雄位置 (x, z) 归一化 | 2 |
| 英雄基础特征 | 剩余宝箱占比 | 1 |
| 英雄基础特征 | 步数进度 | 1 |
| 英雄基础特征 | 闪现冷却归一化 | 1 |
| 英雄基础特征 | buff 持续时间归一化 | 1 |
| 怪物特征（单个） | 位置 | 2 |
| 怪物特征（单个） | 与英雄距离桶 one-hot | 6 |
| 怪物特征（单个） | 相对方位 one-hot | 9 |
| 怪物特征（单个） | 速度归一化 | 1 |
| 怪物特征（单个） | 是否在视野内 | 1 |
| 宝箱特征（单个） | 位置 | 2 |
| 宝箱特征（单个） | 与英雄距离桶 one-hot | 6 |
| 宝箱特征（单个） | 相对方位 one-hot | 9 |
| buff 特征（单个） | 位置 | 2 |
| buff 特征（单个） | 与英雄距离桶 one-hot | 6 |
| buff 特征（单个） | 相对方位 one-hot | 9 |
| 局部障碍物特征 | 以英雄为中心的 `21 x 21` 局部 `map_info`，障碍物记为 1，可通行记为 0 | 21 x 21 |
| 地图压缩位域 | bit 0-1: 地形 4 态（地图外 / 未探索 / 障碍物 / 可通过） | 41 x 41 |
| 地图压缩位域 | bit 2-3: 重复经过强度 3 态（0.0 / 0.5 / 1.0） | 41 x 41 |
| 地图压缩位域 | bit 4-5: 怪物状态 3 态（无 / 估计位置=0.5 / 视野内=1.0） | 41 x 41 |
| 地图压缩位域 | bit 6: 宝箱记忆（见过且未获取=1） | 41 x 41 |
| 地图压缩位域 | bit 7: buff 记忆（见过且未获取=1） | 41 x 41 |

## Model

当前模型改为“分段特征编码 + 地图 CNN + 融合 MLP + Actor/Critic 双头”结构。

| 模块 | 输入 | 核心结构 | 输出 |
| --- | --- | --- | --- |
| Hero Encoder | 6 维（英雄基础特征） | Linear 6->32, ELU, Linear 32->48 | 48 |
| Monster Set Encoder | 2 个槽位，每槽 19 维 | 共享 MLP(19->32->48) + Masked Max Pooling | 48 |
| Treasure Set Encoder | 10 个槽位，每槽 17 维 | 共享 MLP(17->32->48) + Masked Max Pooling | 48 |
| Buff Set Encoder | 2 个槽位，每槽 17 维 | 共享 MLP(17->32->48) + Masked Max Pooling | 48 |
| Local Obstacle Encoder | `21x21=441` 维局部障碍物平面 | MLP(441->128->48) | 48 |
| Map CNN Encoder | 解压后的 8x41x41 地图特征 | 3 个 IMPALABlock（8->20->32->48）+ AdaptiveAvgPool + MLP(48->256->192, LayerNorm) | 192 |
| Fusion MLP | 拼接后 432 维 | Linear 432->320, LayerNorm, ELU, Linear 320->160, ELU | 160 |
| Actor Head | 160 | Linear 160->ACTION_NUM | 16 |
| Critic Head | 160 | Linear 160->VALUE_NUM | 1 |

| 观测切分 | 维度 |
| --- | --- |
| Hero 基础特征 | 6 |
| Monster 特征总维度（2x19） | 38 |
| Treasure 特征总维度（10x17） | 170 |
| Buff 特征总维度（2x17） | 34 |
| Local Obstacle 特征总维度（21x21） | 441 |
| Map 特征总维度（1x41x41 packed map） | 1681 |
| 总计 | 2370 |

补充说明：

- Set Encoder 支持 [batch, max_num * input_dim] 或 [batch, max_num, input_dim] 两种输入。
- 空槽位通过 mask 参与池化时被屏蔽；全空集合回落为 0 向量。
- obs 中额外加入了局部 `21x21` 障碍物平面，专门供 MLP 编码，强化贴墙场景下的即时避障判断。
- obs 中地图分支以 1 个 `41x41` 压缩平面存储，进入模型后再解码回 8 通道 CNN 输入。
- 由于 `41x41` 为奇数窗口，英雄始终位于地图输入的唯一中心格。
- 当前模型参数量约 `483,093`，预计 FP32 checkpoint 保存大小约 `1.84 MB`。
- 线性层和卷积层使用正交初始化，偏置初始化为 0。

## Reward

| 类型 | 项目 | 计算方式 | 推荐系数/取值 |
| :--- | :--- | :--- | :--- |
| 每步存活 | 存活奖励 | `REW_ALIVE * max_monster_speed * monster_count`（`max_monster_speed` 为 1 或 2，`monster_count` 为当前怪物数量） | `REW_ALIVE = +0.05` |
| 地图探索 | 新开图奖励 | `REW_EXPLORE_MAP * 当前步新探索到的可通过格子数` | `REW_EXPLORE_MAP = 1 / 2000` |
| 核心目标 | 宝箱获取 | `curr.treasures_collected - last.treasures_collected` | `REW_TREASURE = +5.0` |
| 战略资源 | Buff 获取 | `curr.collected_buff - last.collected_buff` | `REW_BUFF = +3.0` |
| 塑形引导 | 靠近最近宝箱 | 仅当最近可见怪物距离 `>= APPROACH_SAFE_MONSTER_DISTANCE` 时，按 `last_dist - current_dist` 给奖励（当前步得到宝箱跳过计算） | `REW_APPROACH = +0.01` |
| 危险回避 | 脱离危险奖励 | 当上一步最近可见怪物距离 `< 6` 且本步最近可见怪物距离增大时，按增大的距离给奖励（上限 3） | `REW_ESCAPE_DANGER = +0.08` |
| 风险约束 | 危险抢箱惩罚 | 拿到宝箱时若最近可见怪物距离 `< 5`，按 `(5 - dist)` 追加惩罚 | `REW_TREASURE_DANGER = -0.2` |
| 动作规范 | 无效移动惩罚 | 撞墙（`last_pos == current_pos` 且没有放技能） | `REW_INVALID_MOVE = -0.5` |
| 动作规范 | 闪现惩罚 | 闪现仅在“位移 > 6 且（穿墙或矩形内穿怪）且与上一步视野怪物距离均增大”时免罚，否则惩罚 | `REW_BAD_FLASH = -2.0` |
| 行为约束 | 重复经过惩罚 | `max(0, sum_repeat_count_5x5 - 25) * REW_REPEAT_SCALE` | `REW_REPEAT_SCALE = -0.06` |
| 行为约束 | 拖延惩罚 | 近 15 步位移 `< 6` 时惩罚；拿到宝箱或 buff 后重置窗口 | `REW_PROCRASTINATION = -0.1` |
| 终局奖励 | 被抓/失败 | 任务失败（最后一帧） | `REW_FAIL = -10.0` |
| 终局奖励 | 存活到上限 | 达到最大步数完美通关（最后一帧） | `REW_WIN = 0.0` |

补充说明：

- 每步 reward 在 `StateManager.get_reward` 中计算，最后统一乘以 `REW_GLOBAL_SCALE`。
- 终局奖励在 `workflow` 中仅加到最后一帧样本（`collector[-1].reward += final_reward`）。
- 首帧没有上一帧状态时，仍会计算存活奖励和新开图奖励。
- 当前存活奖励同时受“当前怪物最大速度”和“当前怪物数量”影响，怪越快、怪越多，存活分量越大。
- 探索奖励只统计“任务地图内、此前未知、当前确认为可通过”的格子，不再给新发现的障碍或边缘无效区域奖励。
- 已删除“怪物逼近惩罚”，避免模型在必须经过怪物附近时学成原地犹豫；风险相关塑形改为“脱险加分 + 危险抢箱扣分”。
- 靠近宝箱塑形增加了风险门控：只有最近可见怪物不太近时才给 `reward_approach`，避免宝箱塑形与避险目标相互冲突。
