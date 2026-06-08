from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.io_utils import load_point_cloud
from src.pipeline import build_bev_maps, cluster_obstacles_dbscan, crop_points, remove_ground_ransac
from src.visualization import save_summary, save_visualization


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LiDAR BEV obstacle detection pipeline.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file", type=Path, help="Path to a single .las, .laz or .bin point cloud file")
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory with .las, .laz or .bin point cloud files to process in batch mode",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs" / "default.json",
        help="Path to a JSON config file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs",
        help="Directory where output artifacts will be saved",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories when --input-dir is used",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process in batch mode",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already have both PNG and JSON outputs",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="batch_summary.json",
        help="Filename for the aggregated batch summary written in batch mode",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Save only JSON summaries and skip PNG rendering",
    )
    return parser.parse_args()


def configure_runtime(output_dir: Path) -> None:
    mpl_config_dir = output_dir / ".mplconfig"
    xdg_cache_dir = output_dir / ".cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))


def process_file(file_path: Path, cfg: dict, output_dir: Path, summary_only: bool = False) -> dict[str, Any]:
    points = load_point_cloud(file_path, max_points=cfg["max_points"])
    points = crop_points(
        points,
        (cfg["x_min"], cfg["x_max"]),
        (cfg["y_min"], cfg["y_max"]),
        (cfg["z_min"], cfg["z_max"]),
    )
    if len(points) == 0:
        raise ValueError(f"No points left after cropping for {file_path}")

    plane, ground_points, nonground_points = remove_ground_ransac(
        points,
        iterations=cfg["ransac_iterations"],
        distance_threshold=cfg["ground_distance_threshold"],
        seed=cfg["seed"],
    )
    _, clusters = cluster_obstacles_dbscan(
        nonground_points,
        eps=cfg["dbscan_eps"],
        min_samples=cfg["dbscan_min_samples"],
        min_cluster_points=cfg["min_cluster_points"],
    )
    density_map, height_map = build_bev_maps(
        points,
        (cfg["x_min"], cfg["x_max"]),
        (cfg["y_min"], cfg["y_max"]),
        cfg["bev_resolution"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = file_path.stem
    image_path = output_dir / f"{stem}_bev_detection.png"
    summary_path = output_dir / f"{stem}_summary.json"

    if not summary_only:
        save_visualization(
            output_path=image_path,
            source_file=file_path,
            points=points,
            ground_points=ground_points,
            nonground_points=nonground_points,
            density_map=density_map,
            height_map=height_map,
            clusters=clusters,
            xlim=(cfg["x_min"], cfg["x_max"]),
            ylim=(cfg["y_min"], cfg["y_max"]),
        )
    save_summary(
        output_path=summary_path,
        source_file=file_path,
        num_points=len(points),
        num_ground=len(ground_points),
        num_nonground=len(nonground_points),
        plane=plane,
        clusters=clusters,
    )

    return {
        "source_file": str(file_path.resolve()),
        "num_points": len(points),
        "num_ground": len(ground_points),
        "num_nonground": len(nonground_points),
        "num_clusters": len(clusters),
        "image_path": None if summary_only else str(image_path.resolve()),
        "summary_path": str(summary_path.resolve()),
    }


def iter_input_files(input_dir: Path, recursive: bool = False) -> list[Path]:
    files: list[Path] = []
    iterator_name = "rglob" if recursive else "glob"
    for pattern in ("*.las", "*.laz", "*.bin"):
        files.extend(sorted(getattr(input_dir, iterator_name)(pattern)))
    return files


def should_skip(file_path: Path, output_dir: Path) -> bool:
    stem = file_path.stem
    image_path = output_dir / f"{stem}_bev_detection.png"
    summary_path = output_dir / f"{stem}_summary.json"
    return image_path.exists() and summary_path.exists()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_runtime(args.output_dir)

    if args.file is not None:
        result = process_file(args.file, cfg, args.output_dir, summary_only=args.summary_only)
        print(f"Input points: {result['num_points']}")
        print(f"Ground points: {result['num_ground']}")
        print(f"Non-ground points: {result['num_nonground']}")
        print(f"Obstacle clusters: {result['num_clusters']}")
        if result["image_path"] is not None:
            print(f"Saved visualization: {result['image_path']}")
        print(f"Saved summary: {result['summary_path']}")
        return

    input_files = iter_input_files(args.input_dir, recursive=args.recursive)
    if not input_files:
        raise SystemExit(f"No .las, .laz or .bin files found in {args.input_dir}")
    if args.max_files is not None:
        input_files = input_files[: args.max_files]

    results: list[dict[str, Any]] = []
    print(f"Batch mode: processing {len(input_files)} files from {args.input_dir}")
    for index, file_path in enumerate(input_files, start=1):
        if args.skip_existing and should_skip(file_path, args.output_dir):
            print(f"[{index}/{len(input_files)}] Skipping {file_path.name} (outputs already exist)")
            continue
        print(f"[{index}/{len(input_files)}] Processing {file_path.name}")
        result = process_file(file_path, cfg, args.output_dir, summary_only=args.summary_only)
        results.append(result)
        print(
            f"  points={result['num_points']} ground={result['num_ground']} "
            f"nonground={result['num_nonground']} clusters={result['num_clusters']}"
        )

    batch_summary_path = args.output_dir / args.summary_name
    batch_summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved batch summary: {batch_summary_path}")


if __name__ == "__main__":
    main()
