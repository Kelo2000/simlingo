import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from simlingo_training.utils.custom_types import DatasetOutput


def _quat_to_mat(q: List[float]) -> np.ndarray:
    """Convert quaternion [w,x,y,z] to rotation matrix."""
    w, x, y, z = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ], dtype=np.float32)


class Data_NuScenesBehavior(Dataset):
    """Dataset loader for merged NuScenes/DriveLM behaviour data."""

    def __init__(self, data_path: str, split: str = "train", pred_len: int = 11, **_: Optional[dict]) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.split = split
        self.pred_len = pred_len

        self.frames: List[Dict] = []
        split_path = self.data_path / split
        if not split_path.exists():
            raise FileNotFoundError(f"{split_path} does not exist; expected separate '{split}' split")
        for jf in split_path.rglob("*.json"):
            with open(jf, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and "frames" in data:
                self.frames.extend(data["frames"])
            elif isinstance(data, list):
                self.frames.extend(data)
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

        future = np.asarray(frame.get("ego_info", {}).get("future_positions", []), dtype=np.float32)
        if future.size == 0:
            waypoints = np.zeros((self.pred_len, 2), dtype=np.float32)
        else:
            waypoints = future[: self.pred_len, :2]
            if waypoints.shape[0] < self.pred_len:
                pad = np.repeat(waypoints[-1:], self.pred_len - waypoints.shape[0], axis=0)
                waypoints = np.concatenate([waypoints, pad], axis=0)

        d = np.linalg.norm(np.diff(waypoints, axis=0, prepend=waypoints[:1]), axis=1)
        wps_1d = np.stack([np.cumsum(d, dtype=np.float32), np.zeros_like(d)], axis=1)

        cam = frame.get("camera_params", {})
        K = np.asarray(cam.get("camera_intrinsic", []), dtype=np.float32)
        if K.size == 0:
            K_tensor = None
        elif K.shape != (3, 3):
            raise ValueError(f"Expected (3,3) intrinsics, got {K.shape}")
        else:
            K_tensor = torch.tensor(K, dtype=torch.float32)

        rot = cam.get("rotation")
        trans = cam.get("translation")
        if rot is None or trans is None:
            E_tensor = None
        else:
            R = _quat_to_mat(rot)
            E = np.eye(4, dtype=np.float32)
            E[:3, :3] = R
            E[:3, 3] = np.asarray(trans, dtype=np.float32)
            E_tensor = torch.tensor(E, dtype=torch.float32)

        qa_list = frame.get("behavior_qa", [])
        target_cmd = None
        if qa_list:
            qa = qa_list[0]
            target_cmd = qa.get("A", "")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Predict the waypoints."},
                    {"type": "image"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Waypoints:"}],
            },
        ]
        answer = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Waypoints:"}],
            }
        ]

        return DatasetOutput(
            conversation=conversation,
            answer=answer,
            image_ff=image,
            image_ff_org_size=image,
            waypoints=waypoints,
            waypoints_1d=wps_1d,
            path=None,
            target_points=target_cmd,
            speed=frame.get("ego_info", {}).get("speed", 0.0),
            placeholder_values={},
            measurement_path=frame.get("image_path"),
            dataset="nuscenes_behavior",
            camera_intrinsics=K_tensor,
            camera_extrinsics=E_tensor,
            qa_templates=None,
            eval_infos=None,
        )
