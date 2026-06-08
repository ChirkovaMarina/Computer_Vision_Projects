from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from srccam.srccam.calib import Calib
from srccam.srccam.load_calib import CalibReader


def load_point_cloud(file_path: Path, max_points: int | None = None) -> np.ndarray:
    suffix = file_path.suffix.lower()
    if suffix in {".las", ".laz"}:
        try:
            import laspy
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("Missing dependency: laspy[lazrs]") from exc
        las = laspy.read(file_path)
        points = np.column_stack((las.x, las.y, las.z)).astype(np.float32)
    elif suffix == ".bin":
        raw = np.fromfile(file_path, dtype=np.float32)
        if raw.size % 4 == 0:
            points = raw.reshape(-1, 4)[:, :3]
        elif raw.size % 3 == 0:
            points = raw.reshape(-1, 3)
        else:
            raise ValueError(f"Unsupported .bin layout in {file_path}")
    else:
        raise ValueError(f"Unsupported point cloud format: {file_path.suffix}")

    if max_points is not None and len(points) > max_points:
        step = max(1, len(points) // max_points)
        points = points[::step]
    return points


def load_frame(image_file: Path | None, video_file: Path | None, frame_index: int) -> np.ndarray:
    if image_file is not None:
        frame = cv2.imread(str(image_file))
        if frame is None:
            raise FileNotFoundError(f"Failed to read image: {image_file}")
        return frame

    if video_file is None:
        raise ValueError("Either image_file or video_file must be provided")

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_file}")
    if frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_file}")
    return frame


def load_calibration(calib_file: Path) -> Calib:
    params = CalibReader(
        file_name=str(calib_file),
        param=["K", "D", "r", "t"],
    ).read()
    return Calib(params)
