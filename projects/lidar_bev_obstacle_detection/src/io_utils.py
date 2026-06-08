from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import laspy
except ImportError:
    laspy = None


def load_point_cloud(path: Path, max_points: int | None = None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".las", ".laz"}:
        if laspy is None:
            raise SystemExit("Missing dependency: laspy[lazrs]")
        with laspy.open(path) as fh:
            las = fh.read()
        points = np.column_stack((las.x, las.y, las.z)).astype(np.float32)
    elif suffix == ".bin":
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 4 == 0:
            raw = raw.reshape(-1, 4)
            points = raw[:, :3]
        elif raw.size % 3 == 0:
            points = raw.reshape(-1, 3)
        else:
            raise ValueError(f"Unsupported .bin layout in {path}")
        points = points.astype(np.float32)
    else:
        raise ValueError(f"Unsupported point cloud format: {path.suffix}")

    if max_points is not None and len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[idx]
    return points
