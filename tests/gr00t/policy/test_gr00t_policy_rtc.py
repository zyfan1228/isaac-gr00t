from unittest.mock import MagicMock, patch

import numpy as np
import torch

from gr00t.data.types import ModalityConfig


EMBODIMENT = "new_embodiment"
VIDEO_KEY = "cam"
STATE_KEYS = ["right_arm", "right_hand"]
ACTION_KEYS = ["right_arm", "right_hand"]
LANGUAGE_KEY = "task"


def _build_modality_configs():
    return {
        EMBODIMENT: {
            "video": ModalityConfig(delta_indices=[0], modality_keys=[VIDEO_KEY]),
            "state": ModalityConfig(delta_indices=[0], modality_keys=STATE_KEYS),
            "action": ModalityConfig(delta_indices=[0, 1, 2], modality_keys=ACTION_KEYS),
            "language": ModalityConfig(delta_indices=[0], modality_keys=[LANGUAGE_KEY]),
        }
    }


def _make_observation():
    return {
        "video": {
            VIDEO_KEY: np.random.randint(0, 255, (1, 1, 8, 8, 3), dtype=np.uint8),
        },
        "state": {
            "right_arm": np.random.randn(1, 1, 7).astype(np.float32),
            "right_hand": np.random.randn(1, 1, 6).astype(np.float32),
        },
        "language": {
            LANGUAGE_KEY: [["pick up the block"]],
        },
    }


def test_get_action_injects_rtc_action_context_and_forwards_model_options():
    mock_model = MagicMock()
    mock_model.eval = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.get_action = MagicMock(
        return_value={"action_pred": torch.zeros(1, 3, 13, dtype=torch.float32)}
    )

    mock_processor = MagicMock()
    mock_processor.modality_configs = _build_modality_configs()
    mock_processor.get_modality_configs.return_value = _build_modality_configs()
    mock_processor.eval = MagicMock()
    mock_processor.collator = MagicMock(return_value={"dummy": torch.zeros(1)})
    mock_processor.decode_action = MagicMock(
        return_value={
            "right_arm": np.zeros((1, 3, 7), dtype=np.float32),
            "right_hand": np.zeros((1, 3, 6), dtype=np.float32),
        }
    )
    mock_processor.side_effect = lambda messages: {"messages": messages}

    with (
        patch("gr00t.policy.gr00t_policy.AutoModel") as MockAutoModel,
        patch("gr00t.policy.gr00t_policy.AutoProcessor") as MockAutoProcessor,
        patch("pathlib.Path.is_dir", return_value=False),
        patch("pathlib.Path.exists", return_value=True),
        patch.dict("sys.modules", {"gr00t.model": MagicMock()}),
    ):
        MockAutoModel.from_pretrained.return_value = mock_model
        MockAutoProcessor.from_pretrained.return_value = mock_processor

        from gr00t.policy.gr00t_policy import Gr00tPolicy

        policy = Gr00tPolicy(
            embodiment_tag=EMBODIMENT,
            model_path="/fake/path",
            device="cpu",
        )

    observation = _make_observation()
    action_context = {
        "right_arm": np.ones((1, 3, 7), dtype=np.float32),
        "right_hand": np.ones((1, 3, 6), dtype=np.float32) * 2,
    }
    options = {
        "action_context": action_context,
        "action_horizon": 3,
        "rtc_overlap_steps": 3,
        "rtc_frozen_steps": 2,
        "rtc_ramp_rate": 4.0,
    }

    policy.get_action(observation, options=options)

    processor_messages = mock_processor.call_args.args[0]
    vla_step_data = processor_messages[0]["content"]
    assert set(vla_step_data.actions.keys()) == {"right_arm", "right_hand"}
    np.testing.assert_array_equal(vla_step_data.actions["right_arm"], np.ones((3, 7), dtype=np.float32))
    np.testing.assert_array_equal(vla_step_data.actions["right_hand"], np.ones((3, 6), dtype=np.float32) * 2)

    forwarded_options = mock_model.get_action.call_args.kwargs["options"]
    assert forwarded_options == {
        "action_horizon": 3,
        "rtc_overlap_steps": 3,
        "rtc_frozen_steps": 2,
        "rtc_ramp_rate": 4.0,
    }
