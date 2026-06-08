from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from srccam.srccam.calib import Calib


@dataclass
class ProjectionResult:
    points_vehicle: np.ndarray
    points_camera: np.ndarray
    pixels: np.ndarray
    depths: np.ndarray


def crop_points(
    points: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
) -> np.ndarray:
    mask = (
        (points[:, 0] >= xlim[0]) & (points[:, 0] <= xlim[1]) &
        (points[:, 1] >= ylim[0]) & (points[:, 1] <= ylim[1]) &
        (points[:, 2] >= zlim[0]) & (points[:, 2] <= zlim[1])
    )
    return points[mask]


def transform_vehicle_to_camera(points_vehicle: np.ndarray, calib: Calib) -> np.ndarray:
    rotated = (calib.r @ points_vehicle.T) - calib.t
    points_camera = (calib.cam_to_vr @ rotated).T
    return np.asarray(points_camera, dtype=np.float32)


def project_points(
    points_vehicle: np.ndarray,
    calib: Calib,
    image_shape: tuple[int, int, int],
    min_depth: float,
    max_depth: float,
) -> ProjectionResult:
    height, width = image_shape[:2]
    points_camera = transform_vehicle_to_camera(points_vehicle, calib)
    depths = points_camera[:, 2]
    front_mask = (depths >= min_depth) & (depths <= max_depth)
    points_vehicle = points_vehicle[front_mask]
    points_camera = points_camera[front_mask]
    depths = depths[front_mask]
    if len(points_camera) == 0:
        return ProjectionResult(points_vehicle, points_camera, np.empty((0, 2), dtype=np.int32), depths)

    projected = (calib.K @ points_camera.T).T
    uv = projected[:, :2] / projected[:, 2:3]
    inside_mask = (
        (uv[:, 0] >= 0) & (uv[:, 0] < width) &
        (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )
    return ProjectionResult(
        points_vehicle=points_vehicle[inside_mask],
        points_camera=points_camera[inside_mask],
        pixels=np.rint(uv[inside_mask]).astype(np.int32),
        depths=depths[inside_mask],
    )
