# CHANGELOG

obs, reward, model设计细节见[README.md](./README.md)

## 20260425 v2.6

1. 动态低电量阈值重构：删除`LOW_BATTERY_THRESHOLD_BY_CHARGER_BIN`（6桶离散），改用16锚点插值表`LOW_BATTERY_THRESHOLD_INTERP`（0,5,...,75格，`np.interp`按精确距离插值）。dist=0时阈值0.50，dist≈37时≈0.60（对齐v2.4.1有效值），dist=75时0.70
2. 动态边缘清扫系数同步重构：删除`REW_EDGE_CLEAN_BY_CHARGER_BIN`，改用16锚点插值表`REW_EDGE_CLEAN_INTERP`。dist=0时0.050（=v2.4原值），dist≈37时≈0.030，dist=75时0.010，远端信号由原0.005提升至0.010以上
3. 同步删除不再使用的辅助方法`_nearest_charger_distance_bin()`和`_get_bin_config_value()`，新增模块级常量`_CHARGER_DIST_KNOTS`
4. hero特征维度6→7：新增`nearest_charger_dist/MAP_SIZE`连续充电桩距离特征（inf→1.0）；`charge_count`由二值`min(1.0,count)`改为连续归一化`min(count/5.0,1.0)`；`HERO_BASE_FEATURE_DIM=7`，`DIM_OF_OBSERVATION`自动更新为1923

## 20260424 v2.5.1

1. 修改距离桶onehot输出的距离阈值条件，从30一档改为15一档
2. 继续增加智能体靠近官方机器人的违法条件，避免碰撞距离从3改为4
   
## 20260423 v2.5

1. 新增在线旋转数据增强`USE_DATA_AUGMENTATION`，在`workflow/data_augmentation.py`实现`RotationDataAugmentation`。每局随机抽取旋转步数k（从`AUGMENT_ROTATION_STEPS`中均匀采样），推理前将`obs`和`legal_action`旋转到旋转坐标系，网络输出动作逆旋转回环境坐标系后再执行，样本以旋转域的`(obs, legal_action, act)`存入buffer
2. 新增按最近充电桩距离桶动态调整奖励：`LOW_BATTERY_THRESHOLD_BY_CHARGER_BIN`与`REW_EDGE_CLEAN_BY_CHARGER_BIN`（6桶，边界与obs距离桶一致），距离越远低电量阈值越高、边缘清扫系数越低，替代原固定`LOW_BATTERY_THRESHOLD`和`REW_EDGE_CLEAN`

## 20260422 v2.4.1

1. 降低`LOW_BATTERY_THRESHOLD`从0.8到0.6
2. 降低`EW_EDGE_CLEAN`从0.05到0.03，增加探索力度，避免围绕单个充电桩清扫导致卡死

## 20260421 v2.4

1. 增加`LOW_BATTERY_THRESHOLD`从0.3到0.8
2. 避免npc碰撞距离从2增大到3
3. `rob_prelim\code\agent_ppo\conf\train_env_conf.toml`充电桩个数下调，从4减少到3，battery_max从200下调至150（第四次天梯）

## 20260420 v2.3

1. 修改`REW_EDGE_CLEAN`计算方式，判断边缘时，避免考虑前两步的清扫格子
2. 第一次充电前，一直给予充电桩接近奖励
3. obs中新增是否完成第一次充电0/1
4. 将避免npc碰撞距离从1增大到2
5. 微调repeat_count的缩放系数，2->3，最大显示三次重叠范围

## 20260419 v2.2

训练127k得分803.68，问题出在`REW_EDGE_CLEAN`会出现斜上+下这种移动方法来刷分，第一次需要优先找到附近充电桩，否则可能因为刷分导致无法找到充电桩

1. 增加边缘清洁奖励`REW_EDGE_CLEAN=0.05`，当前清扫周围如果存在非上一步的已清扫或障碍物格子，则本步奖励增加该分量。
2. 发现地图存储的问题，x是横向，z是纵向，所以存储repeat_count应该为`[z][x]`而非`[x][z]`，已修正。
3. 修改特征中repeat_count直接clip(0,1)，仅考虑是否访问过
4. 优化legal_action，通过是否为障碍物格子直接判断动作是否合法
5. 调整buffersize: `1e4->5e4`，样本消耗比`reverb_samples_per_insert: 5->3`
6. 优化legal_action，与npc距离不能小于2步

## 20260417 v2.1

1. 增加连续清扫奖励与连续未清扫惩罚：`StateManager` 内维护两个独立 `deque`（`_clean_history_buffer` 存每步清扫污渍格数，`_no_clean_history_buffer` 存每步是否未清扫 1/0），容量由 `CLEAN_HISTORY_BUFFER_MAXLEN` 控制。
2. 减低ent的最终值为1e-4，增大降低schedual step步长为1e5
3. 新增train_step到保存/第一次加载模型中

## 22060417 v2

1. 重构obs, rew, model设计, 线性变化lr, ent
2. 打印所有奖励分量, lr, ent到monitor中
3. 删除_maybe_random_shift_action和RND 分支
4. 台式机上环境数少只有8，buffer 变小为 1e5->1e4，学习率也相应变小，batchsize变小2048->1024

