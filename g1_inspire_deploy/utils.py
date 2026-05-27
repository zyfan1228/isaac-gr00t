import math
import os

import numpy as np
from loguru import logger


def interp_connect_nd(
    vec_1,
    vec_2,
    interp_number=None,
    ignore_clip=True,
    gap_thred=10.0,
    max_interp_number=20,
):
    """N-dimensional interpolation connecting two action chunks.

    Args:
        vec_1: First action chunk, shape (vec_len, n)
        vec_2: Second action chunk, shape (vec_len, n)
        interp_number: Number of interpolation points (auto-computed if None)
        ignore_clip: Whether to ignore the gripper dimension (last column) in gap calculation
        gap_thred: Threshold for external gap below which interp_number is clamped
        max_interp_number: Maximum number of interpolation points

    Returns:
        Tuple of (interpolated_full_sequence, interp_number)
    """
    if interp_number is None:
        if ignore_clip:
            gap_1 = (
                0
                if vec_1.shape[0] == 1
                else np.abs(vec_1[1:, :-1] - vec_1[:-1, :-1]).mean(0).max()
            )
            gap_2 = (
                0
                if vec_2.shape[0] == 1
                else np.abs(vec_2[1:, :-1] - vec_2[:-1, :-1]).mean(0).max()
            )
        else:
            gap_1 = (
                0
                if vec_1.shape[0] == 1
                else np.abs(vec_1[1:] - vec_1[:-1]).mean(0).max()
            )
            gap_2 = (
                0
                if vec_2.shape[0] == 1
                else np.abs(vec_2[1:] - vec_2[:-1]).mean(0).max()
            )
        inner_gap_max = max(gap_1, gap_2)

        external_gap_max = (
            np.abs(vec_1[-1, :-1] - vec_2[0, :-1])
            if ignore_clip
            else np.abs(vec_1[-1] - vec_2[0])
        )
        external_gap_max = external_gap_max.max()
        logger.info(f"external_gap_max: {external_gap_max:.4f}")

        interp_number = (
            np.abs(vec_1[-1, :-1] - vec_2[0, :-1]) / inner_gap_max
            if ignore_clip
            else np.abs(vec_1[-1] - vec_2[0]) / inner_gap_max
        )
        interp_number = np.ceil(interp_number).max()
        interp_number = min(np.int32(interp_number), max_interp_number)

        if external_gap_max < gap_thred:
            interp_number = min(interp_number, int(gap_thred) + 1)

    x1 = np.arange(0, len(vec_1))
    x2 = np.arange(len(vec_1) + interp_number, len(vec_1) + interp_number + len(vec_2))
    x = np.concatenate((x1, x2))
    vec_new = []
    for i in range(vec_1.shape[1]):
        f = interp1d(x, np.concatenate((vec_1[:, i], vec_2[:, i])))
        vec_new.append(f(np.arange(0, len(vec_1) + len(vec_2) + interp_number)))
    return np.array(vec_new).T, interp_number


class ActionSmoother:
    """Smooths action chunks to ensure smooth transitions between consecutive inferences."""

    def __init__(
        self,
        action_start: int = 20,
        action_end: int = 40,
        skip_scale: int = 2,
        smooth_tail_len: int = 5,
        smooth_head_len: int = 5,
        use_action_smooth: bool = True,
    ):
        """
        Args:
            action_start: Start index of action window (before skip scaling)
            action_end: End index of action window (before skip scaling)
            skip_scale: Downsampling factor for action chunk
            smooth_tail_len: Number of tail actions to keep for next chunk connection
            smooth_head_len: Number of head actions to use for interpolation
            use_action_smooth: Whether to apply action smoothing
        """
        self.action_start = action_start
        self.action_end = action_end
        self.skip_scale = skip_scale
        self.smooth_tail_len = smooth_tail_len
        self.smooth_head_len = smooth_head_len
        self.use_action_smooth = use_action_smooth

        self.history_action_tail = None

    def clear_memory(self):
        """Clear history for fresh start."""
        self.history_action_tail = None

    def action_smooth(self, input_data: np.ndarray) -> np.ndarray:
        """Apply smoothing to connect history_action_tail with current action chunk.

        Args:
            input_data: Raw action chunk from model, shape (chunk_size, action_dim)

        Returns:
            Smoothed action chunk with interpolated transition at the beginning
        """
        if self.history_action_tail is None:
            self.history_action_tail = input_data[-self.smooth_tail_len :].copy()
            return input_data

        if not self.use_action_smooth:
            return input_data

        # Apply skip scaling to indices
        action_start = self.action_start // self.skip_scale
        action_end = self.action_end // self.skip_scale

        # Downsample input data
        input_data_scaled = input_data[:: self.skip_scale]
        # Extract action window
        input_data_window = input_data_scaled[action_start:action_end]

        # Take head segment for interpolation with history
        current_action_head = input_data_window[0 : self.smooth_head_len]

        # Interpolate to connect history tail with current head
        smooth_actions_, inter_number = interp_connect_nd(
            self.history_action_tail, current_action_head
        )

        # Extract the interpolated transition portion
        smooth_actions = smooth_actions_[
            len(self.history_action_tail) : len(self.history_action_tail)
            + inter_number
            - 1
        ]

        # Concatenate: interpolated_transition + full_current_action
        result = np.concatenate((smooth_actions, input_data_window))

        # Update history tail for next chunk
        self.history_action_tail = result[-self.smooth_tail_len :].copy()
        return result
