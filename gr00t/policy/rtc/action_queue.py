import logging
from threading import Lock

import torch
from torch import Tensor

from gr00t.policy.rtc.configuration_rtc import RTCConfig

logger = logging.getLogger(__name__)


class ActionQueue:
    def __init__(self, cfg: RTCConfig):
        self.queue = None
        self.original_queue = None
        self.lock = Lock()
        self.last_index = 0
        self.cfg = cfg

    def get(self) -> Tensor | None:
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None

            action = self.queue[self.last_index]
            self.last_index += 1
            return action.clone()

    def qsize(self) -> int:
        if self.queue is None:
            return 0
        length = len(self.queue)
        return length - self.last_index

    def empty(self) -> bool:
        if self.queue is None:
            return True

        length = len(self.queue)
        return length - self.last_index <= 0

    def get_action_index(self) -> int:
        return self.last_index

    def get_left_over(self) -> Tensor | None:
        with self.lock:
            if self.original_queue is None:
                return None
            return self.original_queue[self.last_index :]

    def merge(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        real_delay: int,
        action_index_before_inference: int | None = 0,
    ):
        with self.lock:
            self._check_delays(real_delay, action_index_before_inference)

            if self.cfg.enabled:
                self._replace_actions_queue(original_actions, processed_actions, real_delay)
                return

            self._append_actions_queue(original_actions, processed_actions)

    def _replace_actions_queue(self, original_actions: Tensor, processed_actions: Tensor, real_delay: int):
        self.original_queue = original_actions[real_delay:].clone()
        self.queue = processed_actions[real_delay:].clone()

        logger.debug(f"original_actions shape: {self.original_queue.shape}")
        logger.debug(f"processed_actions shape: {self.queue.shape}")
        logger.debug(f"real_delay: {real_delay}")

        self.last_index = 0

    def _append_actions_queue(self, original_actions: Tensor, processed_actions: Tensor):
        if self.queue is None:
            self.original_queue = original_actions.clone()
            self.queue = processed_actions.clone()
            return

        self.original_queue = torch.cat([self.original_queue, original_actions.clone()])
        self.original_queue = self.original_queue[self.last_index :]

        self.queue = torch.cat([self.queue, processed_actions.clone()])
        self.queue = self.queue[self.last_index :]

        self.last_index = 0

    def _check_delays(self, real_delay: int, action_index_before_inference: int | None = None):
        if action_index_before_inference is None:
            return

        indexes_diff = self.last_index - action_index_before_inference
        if indexes_diff != real_delay:
            logger.warning(
                f"[ACTION_QUEUE] Indexes diff is not equal to real delay. "
                f"Indexes diff: {indexes_diff}, real delay: {real_delay}"
            )
