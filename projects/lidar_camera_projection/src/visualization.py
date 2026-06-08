from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .projection import ProjectionResult


def depth_to_bgr(depths: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    if len(depths) == 0:
        return np.empty((0, 3), dtype=np.uint8)
    normalized = (depths - min_depth) / max(1e-6, max_depth - min_depth)
    normalized = np.clip(normalized, 0.0, 1.0)
    color_values = np.rint((1.0 - normalized) * 255.0).astype(np.uint8)
    color_map = cv2.applyColorMap(color_values.reshape(-1, 1), cv2.COLORMAP_TURBO)
    return color_map.reshape(-1, 3)


def draw_overlay(
    frame: np.ndarray,
    result: ProjectionResult,
    min_depth: float,
    max_depth: float,
    point_radius: int,
) -> np.ndarray:
    overlay = frame.copy()
    colors = depth_to_bgr(result.depths, min_depth, max_depth)
    order = np.argsort(result.depths)[::-1]
    for idx in order:
        x, y = result.pixels[idx]
        color = tuple(int(c) for c in colors[idx])
        cv2.circle(overlay, (int(x), int(y)), point_radius, color, thickness=-1, lineType=cv2.LINE_AA)
    return overlay


def add_titles(original: np.ndarray, overlay: np.ndarray, title_left: str, title_right: str) -> np.ndarray:
    top_pad = 36
    left_panel = cv2.copyMakeBorder(original, top_pad, 0, 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 18))
    right_panel = cv2.copyMakeBorder(overlay, top_pad, 0, 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 18))
    cv2.putText(left_panel, title_left, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(right_panel, title_right, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
    return np.hstack([left_panel, right_panel])


def draw_depth_legend(image: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    legend_width = 92
    legend = np.full((image.shape[0], legend_width, 3), 18, dtype=np.uint8)
    gradient_height = min(280, image.shape[0] - 80)
    gradient_y0 = 48
    gradient = np.linspace(255, 0, gradient_height, dtype=np.uint8).reshape(-1, 1)
    gradient = np.repeat(gradient, 24, axis=1)
    colorbar = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
    legend[gradient_y0:gradient_y0 + gradient_height, 28:52] = colorbar
    cv2.putText(legend, "Depth", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(legend, f"{min_depth:.1f}m", (8, gradient_y0 + gradient_height + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(legend, f"{max_depth:.1f}m", (8, gradient_y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    return np.hstack([image, legend])


def save_overlay_image(
    output_path: Path,
    frame: np.ndarray,
    result: ProjectionResult,
    min_depth: float,
    max_depth: float,
    point_radius: int,
) -> None:
    overlay = draw_overlay(frame, result, min_depth, max_depth, point_radius)
    combined = add_titles(frame, overlay, "Camera Frame", "LiDAR Projection")
    combined = draw_depth_legend(combined, min_depth, max_depth)
    ok = cv2.imwrite(str(output_path), combined)
    if not ok:
        raise RuntimeError(f"Failed to save overlay image: {output_path}")


def save_summary(
    output_path: Path,
    image_source: str,
    point_cloud_source: str,
    calib_source: str,
    num_input_points: int,
    num_projected_points: int,
    min_depth: float,
    max_depth: float,
) -> None:
    payload = {
        "image_source": image_source,
        "point_cloud_source": point_cloud_source,
        "calibration_source": calib_source,
        "num_input_points": num_input_points,
        "num_projected_points": num_projected_points,
        "min_depth_m": float(min_depth),
        "max_depth_m": float(max_depth),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
