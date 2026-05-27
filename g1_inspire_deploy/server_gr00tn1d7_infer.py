import argparse
import base64
from collections import deque
from dataclasses import asdict, dataclass
import os
import signal
import time
from typing import Any

import cv2
from loguru import logger
import numpy as np
from hong_atom_bridge.communication import rpc_pipe

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
from gr00t.policy.gr00t_policy import Gr00tPolicy


DEFAULT_SERVICE_ID = "vla_model_infer_service"
DEFAULT_EMBODIMENT_TAG = "new_embodiment"
DEFAULT_DEVICE = "cuda:0"

G1_INSPIRE_INITIAL_JOINTS = np.array(
    [
        0.0323,
        0.8915,
        0.7804,
        0.1095,
        0.6203,
        -0.3960,
        0.6044,
        -0.4958,
        -1.0567,
        -0.3222,
        0.3484,
        -0.6698,
        -0.3158,
        -0.7585,
        0.7227,
        0.8635,
        0.9283,
        0.9702,
        0.9152,
        0.4878,
        0.6284,
        0.7390,
        0.8404,
        0.8748,
        1.0000,
        0.7515,
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class LayoutPreset:
    state_slices: dict[str, tuple[int, int]]
    action_slices: dict[str, tuple[int, int]]
    output_dim: int | None = None


LAYOUT_PRESETS = {
    "g1_full": LayoutPreset(
        state_slices={
            "left_arm": (0, 7),
            "right_arm": (7, 14),
            "left_hand": (14, 20),
            "right_hand": (20, 26),
        },
        action_slices={
            "left_arm": (0, 7),
            "right_arm": (7, 14),
            "left_hand": (14, 20),
            "right_hand": (20, 26),
        },
        output_dim=26,
    ),
    "g1_right_arm_only": LayoutPreset(
        state_slices={
            "right_arm": (7, 14),
            "right_hand": (20, 26),
        },
        action_slices={
            "right_arm": (7, 14),
            "right_hand": (20, 26),
        },
        output_dim=26,
    ),
    "generic": LayoutPreset(state_slices={}, action_slices={}, output_dim=None),
}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    
    return value


def _normalize_layout_name(layout_name: str | None) -> str:
    normalized = (layout_name or "auto").strip().lower()
    if normalized not in {"auto", *LAYOUT_PRESETS.keys()}:
        raise ValueError(
            f"不支持的 layout={layout_name}，可选值: auto, {', '.join(LAYOUT_PRESETS.keys())}"
        )
    return normalized


class TemporalModalityBuffer:
    def __init__(self, modality_configs: dict[str, ModalityConfig]):
        self._history_sizes = {
            modality: self._compute_history_size(config.delta_indices)
            for modality, config in modality_configs.items()
            if modality in {"video", "state"}
        }
        self._buffers = {
            modality: {
                key: deque(maxlen=self._history_sizes[modality])
                for key in modality_configs[modality].modality_keys
            }
            for modality in self._history_sizes
        }

    @staticmethod
    def _compute_history_size(delta_indices: list[int]) -> int:
        min_delta = min(delta_indices)
        return max(1, 1 - min(0, min_delta))

    def reset(self) -> None:
        for modality_buffers in self._buffers.values():
            for history in modality_buffers.values():
                history.clear()

    def append(self, modality: str, key: str, value: np.ndarray) -> None:
        self._buffers[modality][key].append(np.asarray(value))

    def stack(self, modality: str, key: str, delta_indices: list[int]) -> np.ndarray:
        history = self._buffers[modality][key]
        if not history:
            raise ValueError(f"{modality}.{key} 的历史缓存为空，无法构建 observation")

        ordered_values = list(history)
        current_index = len(ordered_values) - 1
        stacked = []
        for delta in delta_indices:
            source_index = min(max(current_index + delta, 0), current_index)
            stacked.append(ordered_values[source_index])
        return np.stack(stacked, axis=0)


class G1Gr00tAdapter:
    def __init__(
        self,
        policy: Gr00tPolicy,
        *,
        layout_name: str = "auto",
        g1_initial_joints: np.ndarray | None = None,
    ):
        self.policy = policy
        self.modality_configs = self.policy.get_modality_config()
        self.video_keys = list(self.modality_configs["video"].modality_keys)
        self.state_keys = list(self.modality_configs["state"].modality_keys)
        self.action_keys = list(self.modality_configs["action"].modality_keys)
        self.language_key = self.modality_configs["language"].modality_keys[0]
        self.layout_name = _normalize_layout_name(layout_name)
        self.g1_initial_joints = (
            np.asarray(g1_initial_joints, dtype=np.float32)
            if g1_initial_joints is not None
            else G1_INSPIRE_INITIAL_JOINTS.copy()
        )
        self.buffers = TemporalModalityBuffer(self.modality_configs)

        if self.layout_name == "auto":
            self.layout_name = self._infer_layout_name()
        self.layout = LAYOUT_PRESETS[self.layout_name]

        logger.info(
            "GR00T adapter initialized: layout={}, video_keys={}, state_keys={}, action_keys={}",
            self.layout_name,
            self.video_keys,
            self.state_keys,
            self.action_keys,
        )

    def reset(self) -> dict[str, Any]:
        self.buffers.reset()
        info = self.policy.reset()
        return {"status": "ok", "info": _to_jsonable(info)}

    def ping(self) -> dict[str, str]:
        return {"status": "ok", "message": "server is running"}

    def get_modality_config(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "modality_config": {
                modality: asdict(config) for modality, config in self.modality_configs.items()
            },
        }

    def infer(self, input_data: dict[str, Any]) -> dict[str, Any]:
        try:
            if input_data.get("reset") or input_data.get("reset_buffer") or input_data.get("step") in {0, "0"}:
                reset_result = self.reset()
                has_obs_data = any(
                    input_data.get(k) for k in ("obs_img_b64", "images_b64", "state", "joints_state", "task", "task_desc", "instruction")
                )
                if not has_obs_data:
                    return {**reset_result, "action": [], "action_by_key": {}, "horizon": 0}
            observation = self._build_observation(input_data)
            options = input_data.get("options")

            start_time = time.time()
            action_dict, info = self.policy.get_action(observation, options=options)
            model_time = time.time() - start_time

            flat_action = self._flatten_action(action_dict)

            horizon = len(flat_action)
            chunk_dim = len(flat_action[0]) if horizon > 0 else 0
            logger.info(
                "推理结束 | 模型推理耗时 {:.3f} ms | 生成动作维度 [horizon={}, dim={}]",
                model_time * 1000,
                horizon,
                chunk_dim,
            )

            return {
                "status": "ok",
                "action": flat_action,
                "action_by_key": {
                    key: value.squeeze(0).tolist() for key, value in action_dict.items()
                },
                "action_keys": self.action_keys,
                "horizon": horizon,
                "info": _to_jsonable(info),
            }
        except Exception as exc:
            logger.exception("GR00T 推理失败")
            return {
                "status": "error",
                "message": str(exc),
            }

    def _infer_layout_name(self) -> str:
        state_key_set = set(self.state_keys)
        if state_key_set == {"left_arm", "right_arm", "left_hand", "right_hand"}:
            return "g1_full"
        if state_key_set == {"right_arm", "right_hand"}:
            return "g1_right_arm_only"
        
        return "generic"

    def _build_observation(self, input_data: dict[str, Any]) -> dict[str, Any]:
        task_desc = (
            input_data.get("task")
            or input_data.get("task_desc")
            or input_data.get("instruction")
        )
        if not task_desc:
            raise ValueError("缺失任务描述，需提供 task/task_desc/instruction 字段")

        images_by_key = self._extract_images(input_data)
        states_by_key = self._extract_states(input_data)

        for key, image in images_by_key.items():
            self.buffers.append("video", key, image)
        for key, state in states_by_key.items():
            self.buffers.append("state", key, state)

        observation = {
            "video": {},
            "state": {},
            "language": {self.language_key: [[str(task_desc)]]},
        }

        for key in self.video_keys:
            delta_indices = self.modality_configs["video"].delta_indices
            stacked_video = self.buffers.stack("video", key, delta_indices)
            observation["video"][key] = stacked_video[np.newaxis, ...].astype(np.uint8)

        for key in self.state_keys:
            delta_indices = self.modality_configs["state"].delta_indices
            stacked_state = self.buffers.stack("state", key, delta_indices)
            observation["state"][key] = stacked_state[np.newaxis, ...].astype(np.float32)

        return observation

    def _extract_images(self, input_data: dict[str, Any]) -> dict[str, np.ndarray]:
        raw_images = input_data.get("images_b64") or input_data.get("obs_img_b64")
        if raw_images is None:
            raise ValueError("缺失图像字段，需提供 obs_img_b64 或 images_b64")

        if isinstance(raw_images, str):
            if len(self.video_keys) != 1:
                raise ValueError(
                    "当前模型需要多个 video key，请使用 images_b64={key: base64} 形式提供图像"
                )
            raw_images = {self.video_keys[0]: raw_images}

        if not isinstance(raw_images, dict):
            raise ValueError("images_b64/obs_img_b64 必须是 base64 字符串或字典")

        decoded = {}
        for key in self.video_keys:
            if key not in raw_images:
                raise ValueError(f"缺失图像 key={key}，当前 payload 仅包含 {list(raw_images.keys())}")
            decoded[key] = self._decode_image(raw_images[key])

        return decoded

    def _extract_states(self, input_data: dict[str, Any]) -> dict[str, np.ndarray]:
        raw_state = input_data.get("state")
        if isinstance(raw_state, dict):
            return self._coerce_state_dict(raw_state)

        joints_state = input_data.get("joints_state")
        if joints_state is None:
            raise ValueError("缺失状态字段，需提供 state 或 joints_state")
        
        return self._split_joint_vector(joints_state)

    def _coerce_state_dict(self, raw_state: dict[str, Any]) -> dict[str, np.ndarray]:
        states_by_key = {}
        for key in self.state_keys:
            if key not in raw_state:
                raise ValueError(f"缺失状态 key={key}，当前 payload 仅包含 {list(raw_state.keys())}")
            states_by_key[key] = np.asarray(raw_state[key], dtype=np.float32)

        return states_by_key

    def _split_joint_vector(self, joints_state: list[float] | np.ndarray) -> dict[str, np.ndarray]:
        joint_vector = np.asarray(joints_state, dtype=np.float32).reshape(-1)

        if self.layout_name == "generic":
            if len(self.state_keys) != 1:
                raise ValueError(
                    "当前 checkpoint 的 state key 不是 G1 预设，且 payload 只提供了 joints_state。"
                    "请改为直接传 state={key: value}。"
                )
            return {self.state_keys[0]: joint_vector}

        states_by_key = {}
        for key in self.state_keys:
            if key not in self.layout.state_slices:
                raise ValueError(f"layout={self.layout_name} 未定义 state key={key} 的切片")
            start, end = self.layout.state_slices[key]
            if end > joint_vector.shape[0]:
                raise ValueError(
                    f"joints_state 长度不足，无法提取 {key}[{start}:{end}]，当前长度={joint_vector.shape[0]}"
                )
            states_by_key[key] = joint_vector[start:end]

        return states_by_key

    def _flatten_action(self, action_dict: dict[str, np.ndarray]) -> list[list[float]]:
        if self.layout.output_dim is not None and all(
            key in self.layout.action_slices for key in self.action_keys
        ):
            return self._flatten_action_with_layout(action_dict)
        
        return self._flatten_action_generic(action_dict)

    def _flatten_action_with_layout(self, action_dict: dict[str, np.ndarray]) -> list[list[float]]:
        first_action = action_dict[self.action_keys[0]]
        batch_size, horizon = first_action.shape[:2]
        if batch_size != 1:
            raise ValueError(f"当前 RPC server 仅支持 batch_size=1，收到 {batch_size}")

        flat_chunk = np.zeros((horizon, self.layout.output_dim), dtype=np.float32)
        if self.layout_name == "g1_right_arm_only":
            flat_chunk[:] = self.g1_initial_joints[np.newaxis, :]

        for key in self.action_keys:
            action_value = np.asarray(action_dict[key], dtype=np.float32)
            if action_value.ndim != 3:
                raise ValueError(f"action[{key}] 期望形状为 (B, T, D)，收到 {action_value.shape}")

            start, end = self.layout.action_slices[key]
            expected_dim = end - start
            if action_value.shape[2] != expected_dim:
                raise ValueError(
                    f"action[{key}] 最后维度应为 {expected_dim}，实际为 {action_value.shape[2]}"
                )
            flat_chunk[:, start:end] = action_value[0]

        return flat_chunk.tolist()

    def _flatten_action_generic(self, action_dict: dict[str, np.ndarray]) -> list[list[float]]:
        first_action = action_dict[self.action_keys[0]]
        batch_size, horizon = first_action.shape[:2]
        if batch_size != 1:
            raise ValueError(f"当前 RPC server 仅支持 batch_size=1，收到 {batch_size}")

        flat_chunk = []
        for step_index in range(horizon):
            step_values = []
            for key in self.action_keys:
                action_value = np.asarray(action_dict[key], dtype=np.float32)
                if action_value.ndim != 3:
                    raise ValueError(f"action[{key}] 期望形状为 (B, T, D)，收到 {action_value.shape}")
                step_values.extend(action_value[0, step_index].tolist())
            flat_chunk.append(step_values)
        
        return flat_chunk

    @staticmethod
    def _decode_image(b64_str: str) -> np.ndarray:
        try:
            img_bytes = base64.b64decode(b64_str)
        except Exception as exc:
            raise ValueError("obs_img_b64 不是合法的 base64 字符串") from exc

        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("图像解码失败，请检查 payload 是否完整")
        # === for debug test ===
        # save_path = f"./debug_frames/decoded_frame_from_client.png"
        # cv2.imwrite(save_path, img_bgr)
        # === test 结束 ===

        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


class Gr00tRpcInferenceServer:
    def __init__(
        self,
        *,
        service_id: str,
        model_path: str,
        embodiment_tag: str,
        device: str,
        strict: bool,
        layout_name: str,
    ):
        logger.info("Loading GR00T policy from {}", model_path)
        self.policy = Gr00tPolicy(
            model_path=model_path,
            embodiment_tag=EmbodimentTag.resolve(embodiment_tag),
            device=device,
            strict=strict,
        )
        self.adapter = G1Gr00tAdapter(self.policy, layout_name=layout_name)
        self.rpc_server_pipe = rpc_pipe.RPCPipe(service_id, "server")
        self.rpc_server_pipe.registry_server_function("infer", self.adapter.infer)
        self.rpc_server_pipe.registry_server_function("reset", lambda payload: self.adapter.reset())
        self.rpc_server_pipe.registry_server_function("ping", lambda payload: self.adapter.ping())
        self.rpc_server_pipe.registry_server_function(
            "get_modality_config",
            lambda payload: self.adapter.get_modality_config(),
        )

        logger.info(
            "RPC server initialized: service_id={}, endpoints={}",
            service_id,
            ["infer", "reset", "ping", "get_modality_config"],
        )

    def serve_forever(self) -> None:
        logger.info("Server ready. Waiting for RPC requests...")
        try:
            while True:
                signal.pause()
        except AttributeError:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, shutting down server")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy a GR00T N1.7 SFT inference server over hong_atom_bridge RPCPipe."
    )
    parser.add_argument(
        "--service-id",
        default=os.getenv("GR00T_RPC_SERVICE_ID", DEFAULT_SERVICE_ID),
        help="hong_atom_bridge service id",
    )
    parser.add_argument(
        "--model-path",
        default=os.getenv("GR00T_MODEL_PATH"),
        required=os.getenv("GR00T_MODEL_PATH") is None,
        help="Path to the finetuned GR00T checkpoint directory",
    )
    parser.add_argument(
        "--embodiment-tag",
        default=os.getenv("GR00T_EMBODIMENT_TAG", DEFAULT_EMBODIMENT_TAG),
        help="Embodiment tag stored in the checkpoint, usually new_embodiment for custom SFT models",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("GR00T_DEVICE", DEFAULT_DEVICE),
        help="Inference device, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--layout",
        default=os.getenv("GR00T_LAYOUT", "auto"),
        help="Joint layout preset: auto, g1_full, g1_right_arm_only, generic",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("GR00T_STRICT", "true").lower() != "false",
        help="Enable GR00T observation/action validation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"找不到模型路径: {args.model_path}")

    logger.info(
        "Starting hong_atom_bridge GR00T server: service_id={}, model_path={}, embodiment_tag={}, device={}, layout={}, strict={}",
        args.service_id,
        args.model_path,
        args.embodiment_tag,
        args.device,
        args.layout,
        args.strict,
    )

    server = Gr00tRpcInferenceServer(
        service_id=args.service_id,
        model_path=args.model_path,
        embodiment_tag=args.embodiment_tag,
        device=args.device,
        strict=args.strict,
        layout_name=args.layout,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
