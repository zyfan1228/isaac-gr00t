#!/usr/bin/env python

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import DATA_DIR, DEFAULT_EPISODES_PATH, write_info, write_stats, write_tasks
from lerobot.model.kinematics import RobotKinematics


# Default joint name lists and EEF frame names (kept in sync with draw_eef_to_video.py)
LEFT_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

DEFAULT_LEFT_EEF_FRAME_NAME = "left_wrist_yaw_link"
DEFAULT_RIGHT_EEF_FRAME_NAME = "right_wrist_yaw_link"


DEFAULT_STATE_KEY = "observation.state"
DEFAULT_ACTION_KEY = "action"
DEFAULT_STATE_EEF_KEY = "observation.state.eef_pos"
DEFAULT_ACTION_EEF_KEY = "action.eef_pos"

# EEF representation options
EEF_REP_XYZ = "xyz"
EEF_REP_XYZ_ROT6D = "xyz+rot6d"
EEF_REP_XYZ_ROTVEC = "xyz+rotvec"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append absolute dual-arm end-effector position features to a local LeRobot v3.0 dataset. "
            "The script assumes the source state/action joint values are absolute joint positions."
        )
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        required=True,
        help="Path to the local LeRobot v3.0 dataset directory.",
    )
    parser.add_argument(
        "--source_repo_id",
        type=str,
        default=None,
        help="Repo ID used when loading the local source dataset. Defaults to local/<dataset_name>.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help=(
            "Directory for the updated dataset. Defaults to a sibling directory named "
            "<dataset>_with_dual_arm_eef. Mutually exclusive with --inplace."
        ),
    )
    parser.add_argument(
        "--output_repo_id",
        type=str,
        default=None,
        help="Repo ID stored on the generated dataset object. Defaults to <source_repo_id>_with_dual_arm_eef.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Replace the source dataset directory after the new dataset has been generated successfully.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output directory or temporary inplace directory.",
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        required=True,
        help="Path to the robot URDF file used for forward kinematics.",
    )
    parser.add_argument(
        "--left_joint_names",
        type=str,
        default=','.join(LEFT_ARM_JOINT_NAMES),
        help="Comma-separated URDF joint names for the left arm, in FK order. Defaults to canonical G1 names.",
    )
    parser.add_argument(
        "--right_joint_names",
        type=str,
        default=','.join(RIGHT_ARM_JOINT_NAMES),
        help="Comma-separated URDF joint names for the right arm, in FK order. Defaults to canonical G1 names.",
    )
    parser.add_argument(
        "--state_left_joint_indices",
        type=str,
        default=None,
        help=(
            "Optional comma-separated indices of the left-arm joints inside the state vector. "
            "If omitted, the script resolves indices from state feature names using --left_joint_names."
        ),
    )
    parser.add_argument(
        "--state_right_joint_indices",
        type=str,
        default=None,
        help=(
            "Optional comma-separated indices of the right-arm joints inside the state vector. "
            "If omitted, the script resolves indices from state feature names using --right_joint_names."
        ),
    )
    parser.add_argument(
        "--action_left_joint_indices",
        type=str,
        default=None,
        help=(
            "Optional comma-separated indices of the left-arm joints inside the action vector. "
            "If omitted, the script resolves indices from action feature names using --left_joint_names."
        ),
    )
    parser.add_argument(
        "--action_right_joint_indices",
        type=str,
        default=None,
        help=(
            "Optional comma-separated indices of the right-arm joints inside the action vector. "
            "If omitted, the script resolves indices from action feature names using --right_joint_names."
        ),
    )
    parser.add_argument(
        "--left_eef_frame_name",
        type=str,
        default=DEFAULT_LEFT_EEF_FRAME_NAME,
        help=f"URDF frame name of the left-arm end effector (default: {DEFAULT_LEFT_EEF_FRAME_NAME}).",
    )
    parser.add_argument(
        "--right_eef_frame_name",
        type=str,
        default=DEFAULT_RIGHT_EEF_FRAME_NAME,
        help=f"URDF frame name of the right-arm end effector (default: {DEFAULT_RIGHT_EEF_FRAME_NAME}).",
    )
    parser.add_argument(
        "--state_key",
        type=str,
        default=DEFAULT_STATE_KEY,
        help=f"Dataset feature key that stores the full state joint vector. Default: {DEFAULT_STATE_KEY}.",
    )
    parser.add_argument(
        "--action_key",
        type=str,
        default=DEFAULT_ACTION_KEY,
        help=f"Dataset feature key that stores the full action joint vector. Default: {DEFAULT_ACTION_KEY}.",
    )
    parser.add_argument(
        "--state_eef_key",
        type=str,
        default=DEFAULT_STATE_EEF_KEY,
        help=f"New feature key used to store state EEF positions. Default: {DEFAULT_STATE_EEF_KEY}.",
    )
    parser.add_argument(
        "--action_eef_key",
        type=str,
        default=DEFAULT_ACTION_EEF_KEY,
        help=f"New feature key used to store action EEF positions. Default: {DEFAULT_ACTION_EEF_KEY}.",
    )
    parser.add_argument(
        "--joint_unit",
        choices=["radian", "degree"],
        default="radian",
        help="Unit used by the source state/action joint vectors. Default: radian.",
    )
    parser.add_argument(
        "--eef_representation",
        choices=[EEF_REP_XYZ, EEF_REP_XYZ_ROT6D, EEF_REP_XYZ_ROTVEC],
        default=EEF_REP_XYZ,
        help=(
            "EEF representation to store for each arm. "
            "Options: 'xyz' (position only), 'xyz+rot6d' (position + 6D rotation), "
            "'xyz+rotvec' (position + rotation vector). Default: xyz."
        ),
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Python logging level. Default: INFO.",
    )
    return parser.parse_args()


def parse_csv_strings(raw_value: str) -> list[str]:
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    return values


def parse_optional_csv_ints(raw_value: str | None) -> list[int] | None:
    if raw_value is None:
        return None
    indices = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not indices:
        raise ValueError("Provided joint index list is empty.")
    return [int(item) for item in indices]


def derive_local_repo_id(dataset_path: Path) -> str:
    return f"local/{dataset_path.name}"


def derive_output_path(dataset_path: Path, explicit_output_path: Path | None, inplace: bool) -> Path:
    if inplace:
        if explicit_output_path is not None:
            raise ValueError("--output_path cannot be used together with --inplace.")
        return dataset_path.parent / f".{dataset_path.name}_dual_arm_eef_tmp"

    if explicit_output_path is not None:
        return explicit_output_path

    return dataset_path.parent / f"{dataset_path.name}_with_dual_arm_eef"


def prepare_output_path(output_path: Path, force: bool) -> None:
    if not output_path.exists():
        return

    if not force:
        raise FileExistsError(f"Output path already exists: {output_path}")

    if output_path.is_dir():
        shutil.rmtree(output_path)
    else:
        output_path.unlink()


def validate_output_location(dataset_path: Path, output_path: Path) -> None:
    resolved_dataset_path = dataset_path.resolve()
    resolved_output_path = output_path.resolve()

    if resolved_output_path == resolved_dataset_path:
        raise ValueError(
            "Output path must be different from the source dataset path. "
            "Use --inplace if you want to replace the source dataset."
        )

    if resolved_dataset_path in resolved_output_path.parents:
        raise ValueError(
            "Output path cannot be placed inside the source dataset directory. "
            "Choose a sibling or unrelated directory instead."
        )


def normalize_vector_feature_names(feature_info: dict[str, Any], feature_key: str) -> list[str] | None:
    names = feature_info.get("names")
    if names is None:
        return None

    if isinstance(names, tuple):
        names = list(names)

    if isinstance(names, list) and all(isinstance(item, str) for item in names):
        return names

    if (
        isinstance(names, list)
        and len(names) == 1
        and isinstance(names[0], list)
        and all(isinstance(item, str) for item in names[0])
    ):
        return names[0]

    raise ValueError(
        f"Feature '{feature_key}' does not expose a 1D names list. Provide explicit joint indices instead."
    )


def validate_vector_feature(feature_info: dict[str, Any], feature_key: str) -> int:
    shape = tuple(feature_info.get("shape", []))
    if len(shape) != 1:
        raise ValueError(
            f"Feature '{feature_key}' must be 1D to resolve joint indices, but its shape is {shape}."
        )
    return int(shape[0])


def urdf_to_k_pascal(urdf_name: str) -> str:
    """Convert a URDF-style snake_case joint name to dataset style 'kPascalCase',
    e.g. left_shoulder_pitch_joint -> kLeftShoulderPitch
    """
    name = urdf_name
    if name.endswith("_joint"):
        name = name[: -len("_joint")]
    parts = [p for p in name.split("_") if p]
    pascal = "".join(part.capitalize() for part in parts)
    return f"k{pascal}"


def urdf_to_pascal(urdf_name: str) -> str:
    """Convert snake_case -> PascalCase without leading 'k'.
    e.g. left_shoulder_pitch_joint -> LeftShoulderPitch
    """
    name = urdf_name
    if name.endswith("_joint"):
        name = name[: -len("_joint")]
    parts = [p for p in name.split("_") if p]
    return "".join(part.capitalize() for part in parts)


def urdf_strip_joint(urdf_name: str) -> str:
    """Strip trailing '_joint' if present: left_shoulder_pitch_joint -> left_shoulder_pitch"""
    if urdf_name.endswith("_joint"):
        return urdf_name[: -len("_joint")]
    return urdf_name


def rotation_matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a rotation vector (axis * angle).

    Uses a numerically-stable method: for small angles returns zero vector,
    otherwise computes axis from the skew-symmetric part.
    """
    # Ensure shape
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3")

    trace = np.trace(R)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))

    if np.isclose(theta, 0.0):
        return np.zeros(3, dtype=np.float32)

    sin_theta = np.sin(theta)
    if np.isclose(sin_theta, 0.0):
        # theta is close to pi; fallback to extracting axis from diagonal
        # Find the largest diagonal element
        diag = np.array([R[0, 0], R[1, 1], R[2, 2]])
        idx = int(np.argmax(diag))
        axis = np.zeros(3, dtype=np.float64)
        axis[idx] = np.sqrt(max(0.0, (diag[idx] - diag[(idx + 1) % 3] - diag[(idx + 2) % 3] + 1.0) / 2.0))
        # Try to recover sign from off-diagonals
        if not np.isclose(axis[idx], 0.0):
            j = (idx + 1) % 3
            k = (idx + 2) % 3
            axis[j] = (R[j, idx] + R[idx, j]) / (2.0 * axis[idx])
            axis[k] = (R[k, idx] + R[idx, k]) / (2.0 * axis[idx])
        axis = axis / np.linalg.norm(axis)
        return (axis * theta).astype(np.float32)

    # Standard case
    rx = (R[2, 1] - R[1, 2]) / (2.0 * sin_theta)
    ry = (R[0, 2] - R[2, 0]) / (2.0 * sin_theta)
    rz = (R[1, 0] - R[0, 1]) / (2.0 * sin_theta)
    axis = np.array([rx, ry, rz], dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    return (axis * theta).astype(np.float32)


def resolve_feature_joint_indices(
    feature_key: str,
    feature_info: dict[str, Any],
    joint_names: list[str],
    explicit_indices: list[int] | None,
) -> list[int]:
    feature_dim = validate_vector_feature(feature_info, feature_key)

    if explicit_indices is not None:
        if len(explicit_indices) != len(joint_names):
            raise ValueError(
                f"Feature '{feature_key}' received {len(explicit_indices)} explicit indices, "
                f"but {len(joint_names)} joint names were provided for FK."
            )
        resolved_indices = explicit_indices
    else:
        feature_names = normalize_vector_feature_names(feature_info, feature_key)
        if feature_names is None:
            raise ValueError(
                f"Feature '{feature_key}' has no names metadata. Provide explicit indices for this feature."
            )

        index_by_name = {name: index for index, name in enumerate(feature_names)}

        # Try direct name matches first; if missing, attempt common URDF->dataset name variants
        resolved_indices = []
        unmatched: list[str] = []
        for joint_name in joint_names:
            if joint_name in index_by_name:
                resolved_indices.append(index_by_name[joint_name])
                continue

            # candidate variants to try (heuristic)
            candidates = [
                urdf_to_k_pascal(joint_name),
                urdf_to_pascal(joint_name),
                urdf_strip_joint(joint_name),
            ]

            found = False
            for cand in candidates:
                if cand in index_by_name:
                    resolved_indices.append(index_by_name[cand])
                    found = True
                    break

            if not found:
                unmatched.append(joint_name)

        if unmatched:
            # Provide helpful diagnostics showing attempted variants for the missing names.
            variants = {jn: [urdf_to_k_pascal(jn), urdf_to_pascal(jn), urdf_strip_joint(jn)] for jn in unmatched}
            raise ValueError(
                f"Feature '{feature_key}' is missing joint names required for FK: {unmatched}. "
                f"Tried URDF->dataset variants: {variants}. Provide explicit indices (e.g. --state_left_joint_indices) or pass dataset joint names via --left_joint_names/--right_joint_names."
            )

    invalid_indices = [index for index in resolved_indices if index < 0 or index >= feature_dim]
    if invalid_indices:
        raise ValueError(
            f"Feature '{feature_key}' has invalid joint indices {invalid_indices} for a vector of size {feature_dim}."
        )

    return resolved_indices


def create_kinematics_solver(
    urdf_path: str,
    left_joint_names: list[str],
    right_joint_names: list[str],
    left_eef_frame_name: str,
    right_eef_frame_name: str,
) -> tuple[RobotKinematics, RobotKinematics]:
    left_kinematics = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name=left_eef_frame_name,
        joint_names=left_joint_names,
    )
    right_kinematics = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name=right_eef_frame_name,
        joint_names=right_joint_names,
    )
    return left_kinematics, right_kinematics


def stack_vector_column(values: pd.Series, feature_key: str) -> np.ndarray:
    try:
        stacked = np.stack([np.asarray(value, dtype=np.float32) for value in values.to_list()], axis=0)
    except ValueError as error:
        raise ValueError(f"Failed to stack vector data for feature '{feature_key}'.") from error

    if stacked.ndim != 2:
        raise ValueError(
            f"Feature '{feature_key}' must decode to a 2D array of shape (frames, dim), got {stacked.shape}."
        )

    return stacked


def to_degrees(joint_values: np.ndarray, joint_unit: str) -> np.ndarray:
    if joint_unit == "degree":
        return joint_values
    return np.rad2deg(joint_values)


def compute_dual_arm_eef_positions(
    joint_values: np.ndarray,
    left_indices: list[int],
    right_indices: list[int],
    left_kinematics: RobotKinematics,
    right_kinematics: RobotKinematics,
    joint_unit: str,
    progress_label: str,
    eef_representation: str = EEF_REP_XYZ,
) -> np.ndarray:
    # determine rotation per-arm dimension
    if eef_representation == EEF_REP_XYZ:
        rot_dim = 0
    elif eef_representation == EEF_REP_XYZ_ROT6D:
        rot_dim = 6
    elif eef_representation == EEF_REP_XYZ_ROTVEC:
        rot_dim = 3
    else:
        raise ValueError(f"Unsupported eef_representation: {eef_representation}")

    per_arm_dim = 3 + rot_dim
    total_dim = per_arm_dim * 2
    n_frames = joint_values.shape[0]
    eef_positions = np.zeros((n_frames, total_dim), dtype=np.float32)

    left_joint_values_deg = to_degrees(joint_values[:, left_indices], joint_unit)
    right_joint_values_deg = to_degrees(joint_values[:, right_indices], joint_unit)

    for row_index in tqdm(range(n_frames), desc=progress_label, leave=False):
        left_pose = left_kinematics.forward_kinematics(left_joint_values_deg[row_index])
        right_pose = right_kinematics.forward_kinematics(right_joint_values_deg[row_index])

        left_xyz = left_pose[:3, 3].astype(np.float32)
        right_xyz = right_pose[:3, 3].astype(np.float32)

        # left block
        base = 0
        eef_positions[row_index, base: base + 3] = left_xyz
        base += 3

        if rot_dim == 6:
            # use first two columns of rotation matrix (row-major flatten)
            left_r6d = left_pose[:3, :2].reshape(6)
            eef_positions[row_index, base: base + 6] = left_r6d.astype(np.float32)
            base += 6
        elif rot_dim == 3:
            left_rotvec = rotation_matrix_to_rotvec(left_pose[:3, :3])
            eef_positions[row_index, base: base + 3] = left_rotvec
            base += 3

        # right block
        eef_positions[row_index, base: base + 3] = right_xyz
        base += 3

        if rot_dim == 6:
            right_r6d = right_pose[:3, :2].reshape(6)
            eef_positions[row_index, base: base + 6] = right_r6d.astype(np.float32)
            base += 6
        elif rot_dim == 3:
            right_rotvec = rotation_matrix_to_rotvec(right_pose[:3, :3])
            eef_positions[row_index, base: base + 3] = right_rotvec
            base += 3

    return eef_positions


def compute_dataset_eef_features(
    dataset: LeRobotDataset,
    state_key: str,
    action_key: str,
    state_left_indices: list[int],
    state_right_indices: list[int],
    action_left_indices: list[int],
    action_right_indices: list[int],
    left_kinematics: RobotKinematics,
    right_kinematics: RobotKinematics,
    joint_unit: str,
    eef_representation: str = EEF_REP_XYZ,
) -> tuple[np.ndarray, np.ndarray]:
    if eef_representation == EEF_REP_XYZ:
        rot_dim = 0
    elif eef_representation == EEF_REP_XYZ_ROT6D:
        rot_dim = 6
    elif eef_representation == EEF_REP_XYZ_ROTVEC:
        rot_dim = 3
    else:
        raise ValueError(f"Unsupported eef_representation: {eef_representation}")

    per_arm_dim = 3 + rot_dim
    total_dim = per_arm_dim * 2

    state_eef_values = np.zeros((dataset.meta.total_frames, total_dim), dtype=np.float32)
    action_eef_values = np.zeros((dataset.meta.total_frames, total_dim), dtype=np.float32)

    parquet_files = sorted((dataset.root / DATA_DIR).glob("*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet data files found under {dataset.root / DATA_DIR}.")

    for parquet_path in tqdm(parquet_files, desc="Reading data files"):
        frame_df = pd.read_parquet(parquet_path, columns=["index", state_key, action_key])
        dataset_indices = frame_df["index"].to_numpy(dtype=np.int64)

        state_joint_values = stack_vector_column(frame_df[state_key], state_key)
        action_joint_values = stack_vector_column(frame_df[action_key], action_key)

        state_eef_values[dataset_indices] = compute_dual_arm_eef_positions(
            joint_values=state_joint_values,
            left_indices=state_left_indices,
            right_indices=state_right_indices,
            left_kinematics=left_kinematics,
            right_kinematics=right_kinematics,
            joint_unit=joint_unit,
            progress_label=f"FK {parquet_path.name} state",
            eef_representation=eef_representation,
        )
        action_eef_values[dataset_indices] = compute_dual_arm_eef_positions(
            joint_values=action_joint_values,
            left_indices=action_left_indices,
            right_indices=action_right_indices,
            left_kinematics=left_kinematics,
            right_kinematics=right_kinematics,
            joint_unit=joint_unit,
            progress_label=f"FK {parquet_path.name} action",
            eef_representation=eef_representation,
        )

    return state_eef_values, action_eef_values


def sanitize_feature_component(component: str) -> str:
    sanitized = "".join(character if character.isalnum() else "_" for character in component)
    sanitized = sanitized.strip("_")
    return sanitized or "eef"


def build_eef_component_names(left_frame_name: str, right_frame_name: str, eef_representation: str = EEF_REP_XYZ) -> list[str]:
    left_prefix = sanitize_feature_component(left_frame_name)
    right_prefix = sanitize_feature_component(right_frame_name)

    names: list[str] = []
    # left position
    names.extend([f"{left_prefix}_x", f"{left_prefix}_y", f"{left_prefix}_z"])

    # left rotation
    if eef_representation == EEF_REP_XYZ_ROT6D:
        names.extend([f"{left_prefix}_rot6d_{i}" for i in range(6)])
    elif eef_representation == EEF_REP_XYZ_ROTVEC:
        names.extend([f"{left_prefix}_rot_x", f"{left_prefix}_rot_y", f"{left_prefix}_rot_z"])

    # right position
    names.extend([f"{right_prefix}_x", f"{right_prefix}_y", f"{right_prefix}_z"])

    # right rotation
    if eef_representation == EEF_REP_XYZ_ROT6D:
        names.extend([f"{right_prefix}_rot6d_{i}" for i in range(6)])
    elif eef_representation == EEF_REP_XYZ_ROTVEC:
        names.extend([f"{right_prefix}_rot_x", f"{right_prefix}_rot_y", f"{right_prefix}_rot_z"])

    return names


def build_feature_info(component_names: list[str]) -> dict[str, Any]:
    return {
        "dtype": "float32",
        "shape": [len(component_names)],
        "names": [component_names],
    }


def to_object_feature_array(values: np.ndarray) -> np.ndarray:
    object_array = np.empty(values.shape[0], dtype=object)
    object_array[:] = [row for row in values]
    return object_array


class EpisodeRowLoader:
    def __init__(self, dataset: LeRobotDataset):
        self.dataset = dataset
        self._cached_path: Path | None = None
        self._cached_df: pd.DataFrame | None = None

    def load(self, episode_index: int) -> dict[str, Any]:
        episode_info = self.dataset.meta.episodes[episode_index]
        chunk_index = episode_info["meta/episodes/chunk_index"]
        file_index = episode_info["meta/episodes/file_index"]
        parquet_path = self.dataset.root / DEFAULT_EPISODES_PATH.format(
            chunk_index=chunk_index,
            file_index=file_index,
        )

        if self._cached_path != parquet_path:
            self._cached_path = parquet_path
            self._cached_df = pd.read_parquet(parquet_path)

        if self._cached_df is None:
            raise RuntimeError("Episode metadata parquet cache was not initialized.")

        episode_rows = self._cached_df[self._cached_df["episode_index"] == episode_index]
        if episode_rows.empty:
            raise ValueError(f"Episode {episode_index} was not found in {parquet_path}.")

        return episode_rows.iloc[0].to_dict()


class EpisodeTableUpdater:
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root
        self._cached_path: Path | None = None
        self._cached_df: pd.DataFrame | None = None

    def _flush(self) -> None:
        if self._cached_path is None or self._cached_df is None:
            return
        self._cached_df.to_parquet(self._cached_path, index=False)

    def update_episode_stats(
        self,
        parquet_path: Path,
        episode_index: int,
        episode_stats: dict[str, dict[str, np.ndarray]],
    ) -> None:
        if self._cached_path != parquet_path:
            self._flush()
            self._cached_path = parquet_path
            self._cached_df = pd.read_parquet(parquet_path)

        if self._cached_df is None:
            raise RuntimeError("Episode metadata parquet cache was not initialized.")

        row_mask = self._cached_df["episode_index"] == episode_index
        if not row_mask.any():
            raise ValueError(f"Episode {episode_index} was not found in {parquet_path}.")

        for feature_name, feature_stats in episode_stats.items():
            for stat_name, stat_value in feature_stats.items():
                column_name = f"stats/{feature_name}/{stat_name}"
                serialized_value = stat_value.tolist() if isinstance(stat_value, np.ndarray) else stat_value
                # Ensure we assign a value per selected row (broadcast scalar/list to match length)
                n_rows = int(row_mask.sum())
                if n_rows == 0:
                    raise ValueError(f"Episode {episode_index} was not found in {parquet_path}.")
                values = [serialized_value] * n_rows
                # assign using a Series to avoid ndarray broadcasting issues
                self._cached_df.loc[row_mask, column_name] = pd.Series(values, index=self._cached_df.index[row_mask])

    def close(self) -> None:
        self._flush()
        self._cached_path = None
        self._cached_df = None


def normalize_episode_stat_value(value: Any, feature_dtype: str, stat_name: str) -> np.ndarray:
    if feature_dtype in {"image", "video"} and stat_name != "count":
        if isinstance(value, np.ndarray) and value.dtype == object:
            flattened_values = []
            for item in value:
                while isinstance(item, np.ndarray):
                    item = item.flatten()[0]
                flattened_values.append(item)
            return np.array(flattened_values, dtype=np.float64).reshape(3, 1, 1)

        array_value = np.asarray(value)
        if array_value.shape == (3,):
            return array_value.reshape(3, 1, 1)
        return array_value

    array_value = np.asarray(value)
    if array_value.ndim == 0:
        array_value = np.atleast_1d(array_value)
    return array_value


def extract_episode_stats(episode_row: dict[str, Any], features: dict[str, dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    episode_stats: dict[str, dict[str, np.ndarray]] = {}
    for key, value in episode_row.items():
        if not key.startswith("stats/"):
            continue

        stat_path = key.removeprefix("stats/")
        parts = stat_path.split("/")
        if len(parts) != 2:
            continue

        feature_name, stat_name = parts
        if feature_name not in features:
            continue

        feature_dtype = features[feature_name]["dtype"]
        normalized_value = normalize_episode_stat_value(value, feature_dtype, stat_name)
        episode_stats.setdefault(feature_name, {})[stat_name] = normalized_value

    return episode_stats


def extract_episode_metadata(episode_row: dict[str, Any]) -> dict[str, Any]:
    excluded_keys = {
        "episode_index",
        "tasks",
        "length",
        "dataset_from_index",
        "dataset_to_index",
    }
    return {
        key: value
        for key, value in episode_row.items()
        if key not in excluded_keys and not key.startswith("stats/") and not key.startswith("meta/episodes/")
    }


def ensure_tasks_table(meta: LeRobotDatasetMetadata) -> None:
    if meta.tasks is not None:
        return

    meta.tasks = pd.DataFrame({"task_index": pd.Series(dtype=np.int64)})
    meta.tasks.index = pd.Index([], dtype=str)
    write_tasks(meta.tasks, meta.root)


def rebuild_episode_metadata_and_stats(
    source_dataset: LeRobotDataset,
    target_dataset: LeRobotDataset,
    state_eef_values: np.ndarray,
    action_eef_values: np.ndarray,
    state_eef_key: str,
    action_eef_key: str,
) -> None:
    target_meta = target_dataset.meta
    new_feature_defs = {
        state_eef_key: target_meta.features[state_eef_key],
        action_eef_key: target_meta.features[action_eef_key],
    }
    episode_table_updater = EpisodeTableUpdater(target_meta.root)
    added_episode_stats_list: list[dict[str, dict[str, np.ndarray]]] = []

    for episode_index in tqdm(range(source_dataset.meta.total_episodes), desc="Updating episode stats"):
        source_episode = source_dataset.meta.episodes[episode_index]
        start_index = int(source_episode["dataset_from_index"])
        end_index = int(source_episode["dataset_to_index"])
        added_episode_stats = compute_episode_stats(
            {
                state_eef_key: state_eef_values[start_index:end_index],
                action_eef_key: action_eef_values[start_index:end_index],
            },
            new_feature_defs,
        )
        added_episode_stats_list.append(added_episode_stats)

        target_episode = target_dataset.meta.episodes[episode_index]
        chunk_index = int(target_episode["meta/episodes/chunk_index"])
        file_index = int(target_episode["meta/episodes/file_index"])
        parquet_path = target_meta.root / DEFAULT_EPISODES_PATH.format(
            chunk_index=chunk_index,
            file_index=file_index,
        )
        episode_table_updater.update_episode_stats(
            parquet_path=parquet_path,
            episode_index=episode_index,
            episode_stats=added_episode_stats,
        )

    episode_table_updater.close()

    merged_stats = dict(target_meta.stats or source_dataset.meta.stats or {})
    if added_episode_stats_list:
        merged_stats.update(aggregate_stats(added_episode_stats_list))
    write_stats(merged_stats, target_meta.root)
    target_meta.stats = merged_stats
    target_meta.load_metadata()


def copy_optional_subtasks_file(source_root: Path, target_root: Path) -> None:
    source_subtasks = source_root / "meta/subtasks.parquet"
    if not source_subtasks.exists():
        return

    target_subtasks = target_root / "meta/subtasks.parquet"
    target_subtasks.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_subtasks, target_subtasks)


def finalize_inplace_replacement(source_path: Path, generated_path: Path, force: bool) -> None:
    backup_path = source_path.parent / f".{source_path.name}_dual_arm_eef_backup"

    if backup_path.exists():
        if not force:
            raise FileExistsError(f"Backup path already exists: {backup_path}")
        shutil.rmtree(backup_path)

    shutil.move(str(source_path), str(backup_path))
    try:
        shutil.move(str(generated_path), str(source_path))
    except Exception:
        shutil.move(str(backup_path), str(source_path))
        raise
    else:
        shutil.rmtree(backup_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if not args.dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {args.dataset_path}")
    if not args.dataset_path.is_dir():
        raise NotADirectoryError(f"Dataset path must be a directory: {args.dataset_path}")

    left_joint_names = parse_csv_strings(args.left_joint_names)
    right_joint_names = parse_csv_strings(args.right_joint_names)
    source_repo_id = args.source_repo_id or derive_local_repo_id(args.dataset_path)
    output_path = derive_output_path(args.dataset_path, args.output_path, args.inplace)
    output_repo_id = args.output_repo_id or (
        source_repo_id if args.inplace else f"{source_repo_id}_with_dual_arm_eef"
    )

    state_left_joint_indices = parse_optional_csv_ints(args.state_left_joint_indices)
    state_right_joint_indices = parse_optional_csv_ints(args.state_right_joint_indices)
    action_left_joint_indices = parse_optional_csv_ints(args.action_left_joint_indices)
    action_right_joint_indices = parse_optional_csv_ints(args.action_right_joint_indices)

    validate_output_location(args.dataset_path, output_path)
    prepare_output_path(output_path, args.force)

    logging.info("Loading source dataset from %s", args.dataset_path)
    source_dataset = LeRobotDataset(repo_id=source_repo_id, root=args.dataset_path)

    if args.state_key not in source_dataset.meta.features:
        raise KeyError(f"State feature '{args.state_key}' was not found in the source dataset.")
    if args.action_key not in source_dataset.meta.features:
        raise KeyError(f"Action feature '{args.action_key}' was not found in the source dataset.")
    if args.state_eef_key in source_dataset.meta.features:
        raise ValueError(f"Target feature '{args.state_eef_key}' already exists in the source dataset.")
    if args.action_eef_key in source_dataset.meta.features:
        raise ValueError(f"Target feature '{args.action_eef_key}' already exists in the source dataset.")

    state_left_indices = resolve_feature_joint_indices(
        feature_key=args.state_key,
        feature_info=source_dataset.meta.features[args.state_key],
        joint_names=left_joint_names,
        explicit_indices=state_left_joint_indices,
    )
    state_right_indices = resolve_feature_joint_indices(
        feature_key=args.state_key,
        feature_info=source_dataset.meta.features[args.state_key],
        joint_names=right_joint_names,
        explicit_indices=state_right_joint_indices,
    )
    action_left_indices = resolve_feature_joint_indices(
        feature_key=args.action_key,
        feature_info=source_dataset.meta.features[args.action_key],
        joint_names=left_joint_names,
        explicit_indices=action_left_joint_indices,
    )
    action_right_indices = resolve_feature_joint_indices(
        feature_key=args.action_key,
        feature_info=source_dataset.meta.features[args.action_key],
        joint_names=right_joint_names,
        explicit_indices=action_right_joint_indices,
    )

    logging.info("State left joint indices: %s", state_left_indices)
    logging.info("State right joint indices: %s", state_right_indices)
    logging.info("Action left joint indices: %s", action_left_indices)
    logging.info("Action right joint indices: %s", action_right_indices)

    logging.info("Creating kinematics solvers from %s", args.urdf_path)
    left_kinematics, right_kinematics = create_kinematics_solver(
        urdf_path=args.urdf_path,
        left_joint_names=left_joint_names,
        right_joint_names=right_joint_names,
        left_eef_frame_name=args.left_eef_frame_name,
        right_eef_frame_name=args.right_eef_frame_name,
    )

    logging.info("Computing absolute dual-arm EEF positions from state and action features")
    state_eef_values, action_eef_values = compute_dataset_eef_features(
        dataset=source_dataset,
        state_key=args.state_key,
        action_key=args.action_key,
        state_left_indices=state_left_indices,
        state_right_indices=state_right_indices,
        action_left_indices=action_left_indices,
        action_right_indices=action_right_indices,
        left_kinematics=left_kinematics,
        right_kinematics=right_kinematics,
        joint_unit=args.joint_unit,
        eef_representation=args.eef_representation,
    )

    component_names = build_eef_component_names(
        args.left_eef_frame_name, args.right_eef_frame_name, args.eef_representation
    )
    state_feature_info = build_feature_info(component_names)
    action_feature_info = build_feature_info(component_names)
    new_features = {
        args.state_eef_key: (to_object_feature_array(state_eef_values), state_feature_info),
        args.action_eef_key: (to_object_feature_array(action_eef_values), action_feature_info),
    }

    logging.info("Writing updated dataset to %s", output_path)
    updated_dataset = add_features(
        dataset=source_dataset,
        features=new_features,
        output_dir=output_path,
        repo_id=output_repo_id,
    )

    copy_optional_subtasks_file(source_dataset.root, updated_dataset.root)

    logging.info("Rebuilding episode metadata and dataset statistics")
    rebuild_episode_metadata_and_stats(
        source_dataset=source_dataset,
        target_dataset=updated_dataset,
        state_eef_values=state_eef_values,
        action_eef_values=action_eef_values,
        state_eef_key=args.state_eef_key,
        action_eef_key=args.action_eef_key,
    )

    final_dataset_path = output_path
    if args.inplace:
        logging.info("Replacing source dataset in place")
        finalize_inplace_replacement(args.dataset_path, output_path, args.force)
        final_dataset_path = args.dataset_path

    logging.info("Done. Updated dataset path: %s", final_dataset_path)
    logging.info("Added features: %s, %s", args.state_eef_key, args.action_eef_key)
    logging.info("State EEF shape: %s", updated_dataset.meta.features[args.state_eef_key]["shape"])
    logging.info("Action EEF shape: %s", updated_dataset.meta.features[args.action_eef_key]["shape"])


if __name__ == "__main__":
    main()