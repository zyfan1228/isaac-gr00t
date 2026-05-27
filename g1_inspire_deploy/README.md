
# g1_inspire_deploy 使用说明

本说明文档介绍如何在推理部署容器内启动 g1_inspire_deploy 服务，实现 GR00T N1.7 SFT 模型的 G1 真机推理。

主要流程：
1. 启动 gr00t-n1d7-infer:v1.0 推理部署容器
2. 容器内启动 g1_inspire_deploy 服务，dummy 测试


## 1. 启动 gr00t-n1d7-infer:v1.0 推理部署容器

如尚未构建镜像，请参考 [Dockerfile.hong_atom_inference](../docker/README.md) 进行构建。

假设已在 isaac-gr00t/ 目录下，推荐如下命令启动容器：

```bash
docker run -it \
    --name gr00t_n1d7_infer \
    --network host \
    --gpus all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /data/your_hf_cache:/hf_cache \   # 替换为你的 HF 缓存目录
    -v $(pwd):/workspace \                # 保证当前目录为 isaac-gr00t/
    -v /data/your_ckpts:/workspace/ckpts \ # 替换为你的模型 checkpoint 目录
    -v /workspace/.venv \                 # 使用镜像内已构建好的环境
    gr00t-n1d7-infer:v1.0 \
    bash -c "uv pip install --no-deps -e . && bash"
```

> 启动后会自动进入容器。若容器已在运行，可随时通过 `docker exec -it gr00t_n1d7_infer bash` 进入。

GR00T 模型加载时**必须通过 `"model_name"` 字段加载 VLM 基模**（backbone + processor）。默认值为 `"nvidia/Cosmos-Reason2-2B"`（HF hub ID），默认行为每次启动会联网拉取。

推荐改为本地路径，离线启动零延迟，避免 huggingface 或 hf-mirror 网络问题。步骤如下：

### 3.1 将 Cosmos-Reason2-2B 完整基模拷贝到推理机

Cosmos-Reason2-2B 目录应包含 `config.json`、`tokenizer.json`、`processor_config.json` 等完整文件。确保该目录在推理机上可访问：

```bash
# 示例：放在推理机 /workspace/ckpts/ 下
cp -r /path/to/Cosmos-Reason2-2B /workspace/ckpts/Cosmos-Reason2-2B
```

### 3.2 修改 checkpoint 的 config.json 和 processor_config.json

ckpt 中有**两处**存了 `model_name`，都须改为本地路径：

1. 打开 ` ckpts/your_model_checkpoint/config.json`，将 `"model_name"` 的值从 `"nvidia/Cosmos-Reason2-2B"` 改为容器内路径 `"/workspace/ckpts/Cosmos-Reason2-2B"`
2. 打开 ` ckpts/your_model_checkpoint/processor_config.json`，将 `"model_name"` 的值改为相同路径（位置不同，易遗漏）

> ⚠️ **只改 `config.json` 不够**，`AutoProcessor.from_pretrained` 读的是 `processor_config.json`。两处必须一致。
>
> `model_name` 路径**必须**包含 Cosmos-Reason2-2B 的全部文件（权重、config、tokenizer、processor、preprocessor_config）。否则启动时报 `Unsupported model name` 或 `OSError`。


## 2. 容器内启动 g1_inspire_deploy 服务

进入容器并配置好 HF 相关环境变量后，使用如下命令启动推理服务：

```bash
python g1_inspire_deploy/server_gr00tn1d7_infer.py \
    --model-path /workspace/ckpts/your_model_checkpoint \
    --embodiment-tag new_embodiment \
    --device cuda:0 \
    --layout auto
```

**主要参数说明：**

- `--model-path`：必填，finetune 后的 GR00T checkpoint 目录（HuggingFace 格式）。
- `--embodiment-tag`：通常为 `new_embodiment`，需与训练时一致。
- `--device`：推理设备，如 `cuda:0`。
- `--layout`：关节布局，`auto`（自动识别，推荐），`g1_full`（全身26维），`g1_right_arm_only`（右臂+右手13维，输出补齐为26维），`generic`（自定义/单分量）。
- `--service-id`：RPC 服务名，默认 `vla_model_infer_service`。
- `--strict/--no-strict`：是否启用输入输出严格校验。

可通过 `-h/--help` 查看全部参数：

```bash
python g1_inspire_deploy/server_gr00tn1d7_infer.py --help
```

> **服务启动后，默认监听 hong_atom_bridge RPCPipe，暴露 infer/reset/ping/get_modality_config 四个端点。**
>
> 推理服务启动日志会显示模型加载、布局、端点等关键信息。

---

### 客户端调用说明

客户端通过 hong_atom_bridge 的 RPCPipe 发送推理请求，payload 需包含：

- `obs_img_b64` 或 `images_b64`：相机图像，base64 编码（单相机可用字符串，多相机用字典）。
- `joints_state` 或 `state`：机器人当前关节状态（list 或 dict）。
- `task`/`task_desc`/`instruction`：任务描述文本。

**示例 payload：**
```python
{
    "obs_img_b64": "<base64字符串>",
    "joints_state": [26维或13维关节值],
    "task": "put the red block into box"
}
```

返回内容包含：
- `status`：ok/error
- `action`：动作 chunk（二维数组，shape=[horizon, 26]。右臂模式下左臂/左手自动补初始值）
- `action_by_key`：按 key 拆分的动作
- `action_keys`：动作 key 顺序
- `horizon`：动作步数
- `info`：推理附加信息

⚠️ **注意**：推理服务返回消息的 `action` 字段永远与该模型权重在训练时的设置一致。即若模型 SFT 使用的是相对值，则 `action` 中对应也是相对值，client 端需要适配！

如需自定义 observation/action 结构或有特殊部署需求，可修改 `G1Gr00tAdapter` 适配层。

> 启动下面脚本可以执行 dummy 测试。该脚本会加载 lerobot 数据集启动 client，将数据集数据打包为请求发给 server。请确保该脚本在 lerobot 环境下执行。
> ```bash
> # 在 lerobot 环境下执行
> python isaac-gr00t/g1_inspire_deploy/test/g1_client.py
> ```