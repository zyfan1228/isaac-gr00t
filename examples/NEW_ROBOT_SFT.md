# N1D7 自定义本体 SFT 教程

本教程介绍如何在自定义本体上进行 GR00T 模型的监督微调。

## 📋 目录

- [数据准备](#1-数据准备)
- [本体 Config 脚本配置与注册](#2-本体-config-脚本配置与注册)
- [更新数据集 stats.json](#3-根据本体-config-更新数据集-statsjson)
- [启动训练](#4-启动训练)

---

## 1. 数据准备

### 1.1 数据集版本转换

GR00T 数据集基于 LeRobot Dataset v2.1 版本。若本地为 v3.0 版本，需按以下步骤转换：

> ⚠️ **注意**：数据转换需在 lerobot 环境中进行

请参考 [lerobot_conversion](../scripts/lerobot_conversion/README.md) 中的说明进行版本转换。

### 1.2 配置 modality.json

在转换后数据集目录的 `meta/` 路径下添加 `modality.json` 文件，用于映射数据集的模态配置，包括 state、action、video、annotation 等。

<details>
<summary><strong>📖 modality.json 编写规范</strong></summary>

> 详细规范请参考 [data_preparation.md](../getting_started/data_preparation.md) 的 **GR00T LeRobot Specific Requirements** 章节。

**配置示例：**

```json
{
  "state": {
    "left_arm": {"original_key": "observation.state", "start": 0, "end": 7, "absolute": true},
    "right_arm": {"original_key": "observation.state", "start": 7, "end": 14, "absolute": true},
    "left_hand": {"original_key": "observation.state", "start": 14, "end": 20, "absolute": true},
    "right_hand": {"original_key": "observation.state", "start": 20, "end": 26, "absolute": true}
  },
  "action": {
    "left_arm": {"original_key": "action", "start": 0, "end": 7, "absolute": true},
    "right_arm": {"original_key": "action", "start": 7, "end": 14, "absolute": true},
    "left_hand": {"original_key": "action", "start": 14, "end": 20, "absolute": true},
    "right_hand": {"original_key": "action", "start": 20, "end": 26, "absolute": true}
  },
  "video": {
    "ego_view": {
      "original_key": "observation.images.cam_left_high"
    }
  },
  "annotation": {
    "human.task_description": {
      "original_key": "task_index"
    }
  }
}
```
</details>

---

## 2. 本体 Modality Config 脚本配置与注册

### 2.1 完整配置脚本结构解析

Modality Config 是一个 Python 字典，定义在独立脚本中（推荐放在 `examples/你的本体名/` 目录下）。整体结构包含四大模态：

```python
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

your_config = {
    "video": ModalityConfig(...),      # 视频观测（相机图像）
    "state": ModalityConfig(...),      # 状态观测（关节位置、力矩等）
    "action": ModalityConfig(...),     # 动作预测配置
    "language": ModalityConfig(...),   # 语言指令（任务描述）
}

register_modality_config(your_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
```

> 💡 可选参数 `sin_cos_embedding_keys` 和 `mean_std_embedding_keys` 用于指定 state 模态的编码方式，详情见 [data_config.md](../../getting_started/data_config.md)。

### 2.2 四大模态配置详解

#### 2.2.1 Video 模态

定义使用哪些相机的图像作为视觉输入：

```python
"video": ModalityConfig(
    delta_indices=[0],  # 采样时间偏移：[0] 表示仅使用当前帧
    modality_keys=[
        "ego_view",    # 必须与 meta/modality.json 中 "video" 下的键名一致
    ],
),
```

**说明**：
- `delta_indices=[0]` 是标准配置，表示只用当前帧图像
- `modality_keys` 中的键名必须与 `meta/modality.json` 中定义的 video 映射键完全匹配

#### 2.2.2 State 模态

定义机器人的本体感知状态（proprioception）：

```python
"state": ModalityConfig(
    delta_indices=[0],  # 当前时刻的状态
    modality_keys=[
        "left_arm",    # 左臂关节位置
        "right_arm",   # 右臂关节位置
        "left_hand",   # 左手状态
        "right_hand",  # 右手状态
    ],
),
```

#### 2.2.3 Action 模态

这是最核心的配置，决定了动作空间的表示方式：

```python
"action": ModalityConfig(
    delta_indices=list(range(0, 16)),  # 预测未来 16 步动作
    modality_keys=[
        "left_arm", 
        "right_arm",
        "left_hand",
        "right_hand",
    ],
    action_configs=[
        # 每个 modality_key 对应一个 ActionConfig，顺序必须一致
        ActionConfig(
            rep=ActionRepresentation.ABSOLUTE,   # 绝对位置控制
            type=ActionType.NON_EEF,             # 关节空间（非末端执行器）
            format=ActionFormat.DEFAULT,         # 默认格式
        ),
        ActionConfig(
            rep=ActionRepresentation.ABSOLUTE,
            type=ActionType.NON_EEF,
            format=ActionFormat.DEFAULT,
        ),
        ActionConfig(
            rep=ActionRepresentation.ABSOLUTE,
            type=ActionType.NON_EEF,
            format=ActionFormat.DEFAULT,
        ),
        ActionConfig(
            rep=ActionRepresentation.ABSOLUTE,
            type=ActionType.NON_EEF,
            format=ActionFormat.DEFAULT,
        ),
    ],
),
```

#### 2.2.4 Language 模态

定义语言指令来源：

```python
"language": ModalityConfig(
    delta_indices=[0],
    modality_keys=["annotation.human.task_description"],  # 任务描述字段
),
```

### 2.3 ActionConfig 三要素详解

每个 ActionConfig 包含三个核心参数和一个可选参数：

| 参数 | 可选值 | 说明 |
|------|--------|------|
| `rep` (Representation) | `ABSOLUTE` / `RELATIVE` / `DELTA` | 动作表示方式 |
| `type` (Action Type) | `NON_EEF` / `EEF` | 关节空间 vs 末端执行器空间 |
| `format` (Action Format) | `DEFAULT` / `XYZ_ROT6D` / `XYZ_ROTVEC` | 末端执行器姿态的旋转表示 |
| `state_key` | `str` 或 `None` | 可选，EEF 类型时指定对应的 state key |

#### 2.3.1 动作表示方式 (rep)

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `ABSOLUTE` | 目标绝对位置 | 关节位置目标值、二进制夹爪控制 |
| `RELATIVE` | 相对当前状态的增量 (delta) | 需要更好泛化能力的场景 |
| `DELTA` | 与 RELATIVE 类似但计算方式不同 | 特定场景使用（我们训练可以不使用） |

#### 2.3.2 动作类型 (type)

| 类型 | 说明 |
|------|------|
| `NON_EEF` | 关节空间控制，直接输出关节角度/位置 |
| `EEF` | 末端执行器空间控制，输出末端位姿 (position + rotation) |

#### 2.3.3 动作格式 (format)

| 格式 | 说明 |
|------|------|
| `DEFAULT` | 使用数据集原始格式 |
| `XYZ_ROT6D` | 位置 + 6D 旋转表示（更稳定，推荐用于 EEF） |
| `XYZ_ROTVEC` | 位置 + 旋转向量 |

#### 2.3.4 state_key（EEF 类型且相对动作时专用）

当 `type=EEF` 时，动作预测的是末端执行器的位姿变化，但相对动作（RELATIVE）需要知道**当前末端执行器的位置**才能计算目标位置。`state_key` 就是用来指定**从哪个 state 模态获取当前 EEF 位姿**的键名。

```python
ActionConfig(
    rep=ActionRepresentation.RELATIVE,
    type=ActionType.EEF,              # 末端执行器空间
    format=ActionFormat.XYZ_ROT6D,    # 6D 旋转表示
    state_key="left_arm",             # 从 left_arm state 获取当前 EEF 位姿。需保证 left_arm state 是 eef 类型
),
```

> ⚠️ 如果使用 EEF 类型但未指定 `state_key`，系统会默认使用同名的 action key 对应的 state。例如 action key 为 `"left_arm_eef"` 时，会尝试从 `state.left_arm_eef` 获取位姿。

### 2.4 完整配置示例：以 G1_Inspire 混合配置为例（NON_EEF + EEF）

以下是一个混合使用关节空间和末端执行器空间的完整配置示例：

```python
# examples/G1_Inspire/g1_inspire_mixed_config.py

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

g1_inspire_mixed_config = {
    # ===== Video 模态 =====
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],  # 对应 meta/modality.json 中 video.ego_view
    ),

    # ===== State 模态 =====
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm",    # 对应 meta/modality.json 中 state.left_arm
            "right_arm",   # 对应 meta/modality.json 中 state.right_arm
            "left_hand",   # 对应 meta/modality.json 中 state.left_hand
            "right_hand",  # 对应 meta/modality.json 中 state.right_hand
        ],
    ),

    # ===== Action 模态（混合 EEF + NON_EEF）=====
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),  # 预测 16 步
        modality_keys=[
            "left_arm", 
            "right_arm",
            "left_hand",
            "right_hand",
        ],
        action_configs=[
            # 左臂：末端执行器空间，相对控制，6D 旋转表示
            # 注意：EEF + RELATIVE 需要显式指定 state_key
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,    # 相对当前 EEF 位姿的增量
                type=ActionType.EEF,                   # 末端执行器空间
                format=ActionFormat.XYZ_ROT6D,         # 6D 旋转表示
                state_key="left_arm",                  # 从 left_arm state 获取当前 EEF 位姿。left_arm state 需要在 modality.json 中定义为 eef 类型！
            ),
            # 右臂：末端执行器空间，相对控制，6D 旋转表示
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="right_arm",
            ),
            # 左手：关节空间，绝对位置控制（二进制夹爪信号）
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,    # 目标绝对位置
                type=ActionType.NON_EEF,              # 关节空间
                format=ActionFormat.DEFAULT,
                # state_key 默认为 None（NON_EEF 不需要）
            ),
            # 右手：关节空间，绝对位置控制
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),

    # ===== Language 模态 =====
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# 注册配置到全局注册表
register_modality_config(g1_inspire_mixed_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
```

### 2.5 相对关节配置示例

如果所有关节都使用相对增量控制（通常有更好的泛化能力），可以参考以下 action_configs 配置（其他部分与 2.4 示例相同）：

```python
"action": ModalityConfig(
    delta_indices=list(range(0, 16)),
    modality_keys=["left_arm", "right_arm", "left_hand", "right_hand"],
    action_configs=[
        # 左臂：关节空间，相对增量控制
        ActionConfig(
            rep=ActionRepresentation.RELATIVE,  # 关键：使用 RELATIVE 而非 ABSOLUTE
            type=ActionType.NON_EEF,
            format=ActionFormat.DEFAULT,
        ),
        # 右臂：关节空间，相对增量控制
        ActionConfig(
            rep=ActionRepresentation.RELATIVE,
            type=ActionType.NON_EEF,
            format=ActionFormat.DEFAULT,
        ),
        # 左手：关节空间，绝对控制
        ActionConfig(
            rep=ActionRepresentation.ABSOLUTE,   # 手部通常用绝对控制
            type=ActionType.NON_EEF,
            format=ActionFormat.DEFAULT,
        ),
        # 右手：关节空间，绝对控制
        ActionConfig(
            rep=ActionRepresentation.ABSOLUTE,
            type=ActionType.NON_EEF,
            format=ActionFormat.DEFAULT,
        ),
    ],
),
```

### 2.6 关键要求与注意事项

> ✅ **modality_keys 必须严格匹配**
>
> 所有 `modality_keys` 必须与 `meta/modality.json` 中映射后的键名完全一致，包括：
> - 拼写大小写
> - 命名空间前缀（如 `annotation.human.`）
> - 下划线/连字符等特殊字符

> ⚠️ **action_configs 顺序一致性**
>
> `action_configs` 列表的顺序必须与 `modality_keys` 列表的顺序完全对应：
> - `action_configs[0]` 控制 `modality_keys[0]` 的动作
> - `action_configs[1]` 控制 `modality_keys[1]` 的动作
> - 以此类推

> ⚠️ **【重要】action chunk 大小说明**
>
> - `delta_indices=list(range(0, 16))` 确定了 **16 步**的 action chunk 大小
> - 这个大小在**训练和推理时必须保持一致**
> - 使用相对值训练时会据此生成 `meta/relative_stats.json` 文件
> - **action chunk 大小一旦确定便不可更改**（不论相对或绝对模式）
> - 如需更改，必须重新生成 `stats.json` 和 `relative_stats.json` 并从头训练

> 💡 **推荐：新本体使用 NEW_EMBODIMENT**
>
> 官方推荐所有新本体的 SFT 都使用 `EmbodimentTag.NEW_EMBODIMENT`，便于管理与迁移

### 2.7 G1_Inspire 可用配置脚本

| 脚本文件 | 配置类型 | 说明 |
|----------|----------|------|
| `g1_inspire_config.py` | 绝对关节位置 (NON_EEF + ABSOLUTE) | 所有关节使用 ABSOLUTE 表示 |
| `g1_inspire_rel_config.py` | 相对关节位置 (NON_EEF + RELATIVE) | 臂关节使用 RELATIVE，手使用 ABSOLUTE |

---

## 3. 根据本体 Config 更新数据集 stats.json

根据编写好的 modality config 计算 `stats.json` 和 `relative_stats.json`。

> ⚠️ **注意**：命令中各参数必须与实际配置严格对应

`stats.json` 和 `relative_stats.json` 会自动生成在 dataset-path 的 `meta/` 目录下：

```bash
python gr00t/data/stats.py \
    --dataset-path your/data/set \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path ./to/your/modality_config.py
```

---

## 4. 启动训练

### 4.1 基本启动命令

```bash
CUDA_VISIBLE_DEVICES=7 python \
    gr00t/experiment/launch_finetune.py \
    --base-model-path /mnt/data/share/model/GR00T/GR00T-N1.7-3B \
    --dataset-path /mnt/data/fanzhuoyao/data_library/grasp_black_bottle_cg_0320 \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path ./examples/G1_Inspire/g1_inspire_config.py \
    --vlm-model-path /mnt/data/share/model/Cosmos-Reason2-2B \
    --num-gpus 1 \
    --output-dir ./outputs_cg/both-arm_abs-joints_16-chunk \
    --save-total-limit 5 \
    --save-steps 2000 \
    --max-steps 10000 \
    --use-wandb \
    --wandb-project your_project_name \
    --global-batch-size 32 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 8
```

example 路径下子文件夹内也有 `run_train.sh` 脚本，用于启动训练。

> ✅ **提示**：传入的 `--dataset-path` 必须与 `--modality-config` 中的配置配对（即 modality-config 要配合正确计算 stats 后的 dataset-path 使用）

> 💡 **`--vlm-model-path` 说明**：指定本地 VLM 基模（Cosmos-Reason2-2B）路径，避免训练启动时联网拉取。若网络可达且需使用默认 hub ID（`nvidia/Cosmos-Reason2-2B`），可省略此参数。

### 4.2 查看完整参数

```bash
python gr00t/experiment/launch_finetune.py --help
```

### 4.3 高级配置

- **不可见参数**（如去噪步数等）：需在 fine-tune config 或模型 config 中手动设置
- **参数优先级**：`命令行参数` > `微调配置（finetune config）` > `基础配置（base config，即模型 config）`

---

## 5. 预设 Embodiment Tag 使用说明（暂未支持完备。可选）

代码仓库中已预设两个 G1-D 本体 Tag，可直接使用 `--embodiment-tag` 传入。

> ✅ **当前 SFT 仍推荐使用 NEW_EMBODIMENT**。自定义 Tag 仅当长期维护/多机器人需要时再启用。

### 5.1 SFT 阶段可用 Tag

| Tag 名称 | Tag 值 | Projector Index | 适用场景 |
|----------|--------|:---------------:|----------|
| `NEW_EMBODIMENT` | `new_embodiment` | 10 | 默认，适合单次/短期 SFT |
| `G1_D_ARMS_JOINTS` | `unitree_g1_d_with_two_arms_joints_only` | 11 | G1-D 双臂关节空间控制 |
| `G1_D_ARMS_EEF` | `unitree_g1_d_with_two_arms_eef_only` | 12 | G1-D 双臂末端执行器空间控制 |

### 5.2 使用方法

与现有流程完全一致，仅替换 `--embodiment-tag` 参数：

```bash
python gr00t/experiment/launch_finetune.py \
    --base-model-path /path/to/GR00T-N1.7-3B \
    --dataset-path /path/to/your/dataset \
    --embodiment-tag G1_D_ARMS_JOINTS \          # 使用预设 Tag
    --modality-config-path ./examples/xxx/xxx_config.py \
    ... # 其余参数不变
```

### 5.3 注意事项

- **Projector Index 决定权重槽**：不同 Tag 使用不同 Index（11/12），权重互不干扰。同一 Index 的 Tag 共享网络参数。
- **共享 Index 的条件**：运动学结构 + 控制空间定义一致才能共享。例如同一个 G1-D robot，只换数据集，可用同一 Tag。
- **何时需要新 Tag**：引入新的机器人本体、同一机器人改用完全不同的控制空间（如 joints → eef）。此时需在代码中新增 `EmbodimentTag` 枚举成员、`EMBODIMENT_TAG_TO_PROJECTOR_INDEX` 映射（选一个未占用的 Index）、并加入 `FINETUNE_ONLY_TAGS` 集合。

