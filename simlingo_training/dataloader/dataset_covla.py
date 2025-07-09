import json
import os
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from simlingo_training.utils.custom_types import DatasetOutput


class Data_CoVLA(Dataset):
    """Dataset loader for the CoVLA dataset."""

    def __init__(self, data_path: str, split: str = "train", pred_len: int = 11, **kwargs) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.split = split
        self.pred_len = pred_len

        self.frames: List[Dict] = []
        split_path = self.data_path / split
        if not split_path.exists():
            raise FileNotFoundError(
                f"{split_path} does not exist; expected separate '{split}' split"
            )
        json_files = list(split_path.rglob("*.json"))
        for jf in json_files:
            with open(jf, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.frames.extend(data)
                elif isinstance(data, dict) and "frames" in data:
                    self.frames.extend(data["frames"])
                else:
                    self.frames.append(data)

    def __len__(self) -> int:
        return len(self.frames)

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis, ...]
        return img

    def __getitem__(self, index: int) -> DatasetOutput:
        frame = self.frames[index]
        image = self._load_image(frame["image_path"])
        caption = frame.get("caption", {}).get("plain_caption", "")
        trajectory = np.array(frame.get("state", {}).get("trajectory", []), dtype=np.float32)
        if trajectory.size == 0:
            waypoints = np.zeros((self.pred_len, 2), dtype=np.float32)
        else:
            waypoints = trajectory[: self.pred_len, :2]
            if waypoints.shape[0] < self.pred_len:
                pad_len = self.pred_len - waypoints.shape[0]
                pad = np.repeat(waypoints[-1:], pad_len, axis=0)
                waypoints = np.concatenate([waypoints, pad], axis=0)
        waypoints_1d = np.cumsum(
            np.linalg.norm(np.diff(waypoints, axis=0, prepend=waypoints[:1]), axis=1)
        ).astype(np.float32)
        waypoints_1d = np.stack([waypoints_1d, np.zeros_like(waypoints_1d)], axis=1)

        state = frame.get("state", {})
        K = np.asarray(state.get("intrinsic_matrix"), dtype=np.float32)
        if K.size == 0:
            K = None
        elif K.shape != (3, 3):
            raise ValueError(f"Expected (3,3) intrinsics, got {K.shape}")
        else:
            K = torch.tensor(K, dtype=torch.float32)

        E = np.asarray(state.get("extrinsic_matrix"), dtype=np.float32)
        if E.size == 0:
            E = None
        else:
            E = torch.tensor(E, dtype=torch.float32)

        conversation_all = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{caption} Predict the waypoints."},
                    {"type": "image"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Waypoints:"}],
            },
        ]
        conversation_answer = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Waypoints:"}],
            }
        ]

        return DatasetOutput(
            conversation=conversation_all,
            answer=conversation_answer,
            image_ff=image,
            image_ff_org_size=image,
            waypoints=waypoints,
            waypoints_1d=waypoints_1d,
            path=None,
            target_points=None,
            speed=frame.get("state", {}).get("vEgo", 0.0),
            placeholder_values={},
            measurement_path=frame.get("image_path"),
            dataset="covla",
            camera_intrinsics=K,
            camera_extrinsics=E,
        )
