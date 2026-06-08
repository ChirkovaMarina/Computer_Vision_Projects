from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .pipeline import ClusterSummary


def _get_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Missing dependency: matplotlib") from exc
    return plt, Rectangle


def save_summary(
    output_path: Path,
    source_file: Path,
    num_points: int,
    num_ground: int,
    num_nonground: int,
    plane: np.ndarray,
    clusters: list[ClusterSummary],
) -> None:
    payload = {
        "source_file": str(source_file),
        "num_points": num_points,
        "num_ground_points": num_ground,
        "num_nonground_points": num_nonground,
        "ground_plane": [float(x) for x in plane],
        "clusters": [
            {
                "cluster_id": cluster.cluster_id,
                "num_points": cluster.num_points,
                "centroid": list(cluster.centroid),
                "bbox": list(cluster.bbox),
            }
            for cluster in clusters
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_visualization(
    output_path: Path,
    source_file: Path,
    points: np.ndarray,
    ground_points: np.ndarray,
    nonground_points: np.ndarray,
    density_map: np.ndarray,
    height_map: np.ndarray,
    clusters: list[ClusterSummary],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    plt, Rectangle = _get_matplotlib()
    fig = plt.figure(figsize=(16, 10))

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    sample = points[:: max(1, len(points) // 50000)]
    ax3d.scatter(sample[:, 0], sample[:, 1], sample[:, 2], c=sample[:, 2], s=0.4, cmap="viridis")
    ax3d.set_title("Input Point Cloud")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    ax_density = fig.add_subplot(2, 2, 2)
    ax_density.imshow(density_map, cmap="gray")
    ax_density.set_title("BEV Density Map")
    ax_density.set_xticks([])
    ax_density.set_yticks([])

    ax_height = fig.add_subplot(2, 2, 3)
    ax_height.imshow(height_map, cmap="magma")
    ax_height.set_title("BEV Height Map")
    ax_height.set_xticks([])
    ax_height.set_yticks([])

    ax_bev = fig.add_subplot(2, 2, 4)
    if len(ground_points):
        ground_sample = ground_points[:: max(1, len(ground_points) // 18000)]
        ax_bev.scatter(ground_sample[:, 0], ground_sample[:, 1], s=0.2, c="tab:green", label="ground")
    if len(nonground_points):
        nonground_sample = nonground_points[:: max(1, len(nonground_points) // 18000)]
        ax_bev.scatter(nonground_sample[:, 0], nonground_sample[:, 1], s=0.2, c="tab:red", label="nonground")

    cmap = plt.cm.get_cmap("tab20", max(1, len(clusters)))
    for idx, cluster in enumerate(clusters):
        x_min, y_min, x_max, y_max, _, _ = cluster.bbox
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor=cmap(idx), linewidth=2)
        ax_bev.add_patch(rect)
        ax_bev.text(cluster.centroid[0], cluster.centroid[1], str(cluster.cluster_id), color=cmap(idx), fontsize=9)

    ax_bev.set_xlim(xlim)
    ax_bev.set_ylim(ylim)
    ax_bev.set_title("Obstacle Clusters in BEV")
    ax_bev.set_xlabel("X")
    ax_bev.set_ylabel("Y")
    ax_bev.legend(loc="upper right")

    fig.suptitle(source_file.name)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
