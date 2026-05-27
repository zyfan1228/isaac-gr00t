# TODO: VLM 本地基模动态传入（方案 A）

## 目标

训练和推理时通过 CLI 参数动态传入本地 VLM 路径，避免每次联网 check `huggingface.co` / `hf-mirror.com`。

**不修改 checkpoint 的 `config.json`**，保持向后兼容。

---

## 状态总览

| 部分 | 状态 | 备注 |
|------|------|------|
| 训练侧（FinetuneConfig + launch_finetune.py） | ✅ **已完成** | 新增 `--vlm-model-path` 参数 |
| `get_backbone_cls` 匹配放宽 | ✅ **已完成** | 支持本地路径（如 `/workspace/ckpts/...`） |
| 推理侧（Gr00tPolicy + server） | ⏳ **待处理** | 当前用手动改 `config.json` 方案（见 README.md） |

---

## ✅ 已完成

### 1. 训练侧：`gr00t/configs/finetune_config.py`

- 新增字段 `vlm_model_path: str | None = None`

### 2. 训练侧：`gr00t/experiment/launch_finetune.py`

- 移除硬编码 `/mnt/data/share/model/Cosmos-Reason2-2B`
- 改为动态解析 `ft_config.vlm_model_path`：
  ```python
  if ft_config.vlm_model_path is not None:
      config.model.model_name = ft_config.vlm_model_path
      if os.path.isdir(ft_config.vlm_model_path):
          config.training.transformers_local_files_only = True
  ```
- 不传 `--vlm-model-path` 时行为不变（默认 hub ID）

### 3. `gr00t/model/gr00t_n1d7/gr00t_n1d7.py`

- `get_backbone_cls` 匹配条件从 `"nvidia/Cosmos-Reason2"` 放宽为 `"Cosmos-Reason2"`，同时支持 hub ID 和本地路径

### 训练使用方式

```bash
python gr00t/experiment/launch_finetune.py \
    --base-model-path /path/to/GR00T-N1.7-3B \
    --dataset-path /path/to/dataset \
    --vlm-model-path /mnt/data/share/model/Cosmos-Reason2-2B \
    ...
```

---

## ⏳ 待处理（推理侧）

### 4. 推理侧：`gr00t/policy/gr00t_policy.py`

- `Gr00tPolicy.__init__` 新增参数 `vlm_model_path: str | None = None`
- 在 `AutoModel.from_pretrained(model_dir)` 之后：
  - 若 `vlm_model_path` 非空，覆写 `model.config.model_name = vlm_model_path`
  - `from_pretrained` 传 `local_files_only=True`

### 5. 推理侧：`gr00t/model/gr00t_n1d7/gr00t_n1d7.py`

- `Gr00tN1d7.__init__` 中 `model_name` 若 detect 为已有本地目录，自动禁用远端（可选增强）

### 6. 推理侧：`g1_inspire_deploy/server_gr00tn1d7_infer.py`

- `parse_args` 新增 `--vlm-model-path`
- 传给 `Gr00tRpcInferenceServer` → `Gr00tPolicy(vlm_model_path=...)`

### 当前推理替代方案（READY）

手动修改 checkpoint 的 `config.json` + `processor_config.json`，见 [g1_inspire_deploy/README.md](./g1_inspire_deploy/README.md#32-修改-checkpoint-的-configjson-和-processor_configjson)。

---

## 向后兼容

| 场景 | 行为 |
|------|------|
| 旧 checkpoint + 不传 `--vlm-model-path` | 走 `config.json` 的 `model_name`，行为不变 |
| 旧 checkpoint + 传 `--vlm-model-path` | 覆写 VLM 路径，本地加载 |
| 旧训练命令（不传 `--vlm-model-path`） | 不触发新逻辑，行为不变 |

---

## 边界情况

- [ ] 训练时 `vlm_model_path` 不存在 → 提前报错
- [ ] 推理时 `vlm_model_path` 不存在 → 提前报错
- [x] `config.json` 中 `model_name` 已为本地路径但无传参 → 自动 detect 本地目录，不联网
- [ ] `HF_HUB_OFFLINE=1` 与 `local_files_only=True` 同时设置时的冗余安全性
