from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClusterSummary:
    cluster_id: int
    num_points: int
    centroid: tuple[float, float, float]
    bbox: tuple[float, float, float, float, float, float]


def crop_points(
    points: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
) -> np.ndarray:
    mask = (
        (xlim[0] <= points[:, 0]) & (points[:, 0] <= xlim[1]) &
        (ylim[0] <= points[:, 1]) & (points[:, 1] <= ylim[1]) &
        (zlim[0] <= points[:, 2]) & (points[:, 2] <= zlim[1])
    )
    return points[mask]


def _fit_plane_from_points(points3: np.ndarray) -> np.ndarray:
    p1, p2, p3 = points3
    normal = np.cross(p2 - p1, p3 - p1)
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        return np.array([0.0, 0.0, 1.0, -p1[2]], dtype=np.float32)
    normal = normal / norm
    d = -np.dot(normal, p1)
    if normal[2] < 0:
        normal = -normal
        d = -d
    return np.array([normal[0], normal[1], normal[2], d], dtype=np.float32)


def _point_plane_distance(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    return np.abs(points @ plane[:3] + plane[3])


def remove_ground_ransac(
    points: np.ndarray,
    iterations: int,
    distance_threshold: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points) < 3:
        raise ValueError("Need at least 3 points for plane fitting")

    rng = np.random.default_rng(seed)
    low_points = points[points[:, 2] <= np.percentile(points[:, 2], 35)]
    if len(low_points) < 3:
        low_points = points

    best_plane = None
    best_inliers = np.zeros(len(points), dtype=bool)
    for _ in range(iterations):
        sample_idx = rng.choice(len(low_points), size=3, replace=False)
        plane = _fit_plane_from_points(low_points[sample_idx])
        if plane[2] < 0.85:
            continue
        inliers = _point_plane_distance(points, plane) < distance_threshold
        if inliers.sum() > best_inliers.sum():
            best_plane = plane
            best_inliers = inliers

    if best_plane is None:
        raise RuntimeError("Failed to estimate ground plane")

    ground_points = points[best_inliers]
    centroid = ground_points.mean(axis=0)
    _, _, vt = np.linalg.svd(ground_points - centroid, full_matrices=False)
    normal = vt[-1]
    normal = normal / np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
    d = -np.dot(normal, centroid)
    plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float32)
    ground_mask = _point_plane_distance(points, plane) < distance_threshold
    return plane, points[ground_mask], points[~ground_mask]


def cluster_obstacles_dbscan(
    nonground_points: np.ndarray,
    eps: float,
    min_samples: int,
    min_cluster_points: int,
) -> tuple[np.ndarray, list[ClusterSummary]]:
    if len(nonground_points) == 0:
        return np.empty(0, dtype=int), []

    labels = _dbscan_xy(nonground_points[:, :2], eps=eps, min_samples=min_samples)
    clusters: list[ClusterSummary] = []
    next_cluster_id = 0
    filtered_labels = np.full_like(labels, -1)

    for label in np.unique(labels):
        if label < 0:
            continue
        cluster_points = nonground_points[labels == label]
        if len(cluster_points) < min_cluster_points:
            continue
        filtered_labels[labels == label] = next_cluster_id
        x_min, y_min, z_min = cluster_points.min(axis=0)
        x_max, y_max, z_max = cluster_points.max(axis=0)
        centroid = cluster_points.mean(axis=0)
        clusters.append(
            ClusterSummary(
                cluster_id=next_cluster_id,
                num_points=len(cluster_points),
                centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
                bbox=(float(x_min), float(y_min), float(x_max), float(y_max), float(z_min), float(z_max)),
            )
        )
        next_cluster_id += 1
    return filtered_labels, clusters


def _region_query(points_xy: np.ndarray, point_index: int, eps: float) -> np.ndarray:
    diff = points_xy - points_xy[point_index]
    dist2 = np.sum(diff * diff, axis=1)
    return np.flatnonzero(dist2 <= eps * eps)


def _dbscan_xy(points_xy: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    num_points = len(points_xy)
    labels = np.full(num_points, -1, dtype=int)
    visited = np.zeros(num_points, dtype=bool)
    cluster_id = 0

    for point_index in range(num_points):
        if visited[point_index]:
            continue

        visited[point_index] = True
        neighbors = _region_query(points_xy, point_index, eps)
        if len(neighbors) < min_samples:
            continue

        labels[point_index] = cluster_id
        seeds = list(neighbors.tolist())
        seed_set = set(seeds)
        cursor = 0

        while cursor < len(seeds):
            neighbor_index = seeds[cursor]
            if not visited[neighbor_index]:
                visited[neighbor_index] = True
                neighbor_neighbors = _region_query(points_xy, neighbor_index, eps)
                if len(neighbor_neighbors) >= min_samples:
                    for extra_index in neighbor_neighbors.tolist():
                        if extra_index not in seed_set:
                            seed_set.add(extra_index)
                            seeds.append(extra_index)

            if labels[neighbor_index] == -1:
                labels[neighbor_index] = cluster_id
            cursor += 1

        cluster_id += 1
    return labels


def build_bev_maps(
    points: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    resolution: float,
) -> tuple[np.ndarray, np.ndarray]:
    width = int(np.ceil((xlim[1] - xlim[0]) / resolution))
    height = int(np.ceil((ylim[1] - ylim[0]) / resolution))
    density = np.zeros((height, width), dtype=np.float32)
    max_height = np.zeros((height, width), dtype=np.float32)

    x_idx = ((points[:, 0] - xlim[0]) / resolution).astype(int)
    y_idx = ((points[:, 1] - ylim[0]) / resolution).astype(int)
    valid = (0 <= x_idx) & (x_idx < width) & (0 <= y_idx) & (y_idx < height)
    x_idx = x_idx[valid]
    y_idx = y_idx[valid]
    z = points[:, 2][valid]
    rows = height - 1 - y_idx

    np.add.at(density, (rows, x_idx), 1)
    np.maximum.at(max_height, (rows, x_idx), z)
    return np.log1p(density), max_height
