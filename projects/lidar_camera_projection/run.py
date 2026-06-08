from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io_utils import load_calibration, load_frame, load_point_cloud
from src.projection import crop_points, project_points
from src.visualization import save_overlay_image, save_summary


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def configure_runtime(output_dir: Path) -> None:
    mpl_config_dir = output_dir / ".mplconfig"
    xdg_cache_dir = output_dir / ".cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project LiDAR points onto a camera image using calibration data.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "default.json")
    parser.add_argument("--image-file", type=Path, default=None, help="Optional camera image file")
    parser.add_argument("--video-file", type=Path, default=None, help="Optional camera video file used to grab one frame")
    parser.add_argument("--frame-index", type=int, default=None, help="Frame index used with --video-file")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--point-cloud", type=Path, default=None, help="Path to one .las, .laz or .bin file")
    input_group.add_argument("--input-dir", type=Path, default=None, help="Directory with .las, .laz or .bin files")
    parser.add_argument("--calib-file", type=Path, default=None, help="Path to camera calibration YAML")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs")
    parser.add_argument("--summary-only", action="store_true", help="Skip image rendering and save only JSON summary")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan subdirectories in batch mode")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of files to process in batch mode")
    parser.add_argument("--skip-existing", action="store_true", help="Skip scans whose PNG and JSON outputs already exist")
    parser.add_argument("--summary-name", type=str, default="batch_summary.json", help="Filename for the aggregated batch summary")
    return parser.parse_args()


def resolve_path(value: str | None) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def iter_input_files(input_dir: Path, recursive: bool = False) -> list[Path]:
    files: list[Path] = []
    iterator_name = "rglob" if recursive else "glob"
    for pattern in ("*.las", "*.laz", "*.bin"):
        files.extend(sorted(getattr(input_dir, iterator_name)(pattern)))
    return files


def should_skip(file_path: Path, output_dir: Path) -> bool:
    stem = file_path.stem
    overlay_path = output_dir / f"{stem}_camera_overlay.png"
    summary_path = output_dir / f"{stem}_projection_summary.json"
    return overlay_path.exists() and summary_path.exists()


def process_point_cloud(
    point_cloud: Path,
    frame,
    calib,
    calib_file: Path,
    image_source: Path | None,
    video_source: Path | None,
    cfg: dict[str, Any],
    output_dir: Path,
    summary_only: bool,
) -> dict[str, Any]:
    points = load_point_cloud(point_cloud, max_points=cfg.get("max_points"))
    points = crop_points(
        points,
        (cfg["x_min"], cfg["x_max"]),
        (cfg["y_min"], cfg["y_max"]),
        (cfg["z_min"], cfg["z_max"]),
    )
    if len(points) == 0:
        raise ValueError(f"No points left after cropping for {point_cloud}")

    result = project_points(
        points_vehicle=points,
        calib=calib,
        image_shape=frame.shape,
        min_depth=float(cfg["min_depth"]),
        max_depth=float(cfg["max_depth"]),
    )

    stem = point_cloud.stem
    overlay_path = output_dir / f"{stem}_camera_overlay.png"
    summary_path = output_dir / f"{stem}_projection_summary.json"

    if not summary_only:
        save_overlay_image(
            output_path=overlay_path,
            frame=frame,
            result=result,
            min_depth=float(cfg["min_depth"]),
            max_depth=float(cfg["max_depth"]),
            point_radius=int(cfg["point_radius"]),
        )
    save_summary(
        output_path=summary_path,
        image_source=str(image_source if image_source is not None else video_source),
        point_cloud_source=str(point_cloud),
        calib_source=str(calib_file),
        num_input_points=len(points),
        num_projected_points=len(result.pixels),
        min_depth=float(cfg["min_depth"]),
        max_depth=float(cfg["max_depth"]),
    )
    return {
        "point_cloud_source": str(point_cloud.resolve()),
        "num_input_points": len(points),
        "num_projected_points": len(result.pixels),
        "overlay_path": None if summary_only else str(overlay_path.resolve()),
        "summary_path": str(summary_path.resolve()),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_runtime(output_dir)

    image_file = args.image_file or resolve_path(cfg.get("image_file"))
    video_file = args.video_file or resolve_path(cfg.get("video_file"))
    frame_index = args.frame_index if args.frame_index is not None else int(cfg.get("frame_index", 0))
    point_cloud = args.point_cloud if args.point_cloud is not None else resolve_path(cfg.get("point_cloud"))
    input_dir = args.input_dir if args.input_dir is not None else resolve_path(cfg.get("input_dir"))
    if args.input_dir is not None:
        point_cloud = None
    if args.point_cloud is not None:
        input_dir = None
    calib_file = args.calib_file or resolve_path(cfg["calib_file"])

    frame = load_frame(image_file=image_file, video_file=video_file, frame_index=frame_index)
    calib = load_calibration(calib_file)

    if point_cloud is not None:
        processed = process_point_cloud(
            point_cloud=point_cloud,
            frame=frame,
            calib=calib,
            calib_file=calib_file,
            image_source=image_file,
            video_source=video_file,
            cfg=cfg,
            output_dir=output_dir,
            summary_only=args.summary_only,
        )
        print(f"Input points after crop: {processed['num_input_points']}")
        print(f"Projected points inside image: {processed['num_projected_points']}")
        if processed["overlay_path"] is not None:
            print(f"Saved overlay: {processed['overlay_path']}")
        print(f"Saved summary: {processed['summary_path']}")
        return

    if input_dir is None:
        raise SystemExit("Provide either --point-cloud or --input-dir, or set one of them in the config")

    input_files = iter_input_files(input_dir, recursive=args.recursive)
    if not input_files:
        raise SystemExit(f"No .las, .laz or .bin files found in {input_dir}")
    if args.max_files is not None:
        input_files = input_files[: args.max_files]

    results: list[dict[str, Any]] = []
    print(f"Batch mode: processing {len(input_files)} files from {input_dir}")
    for index, file_path in enumerate(input_files, start=1):
        if args.skip_existing and should_skip(file_path, output_dir):
            print(f"[{index}/{len(input_files)}] Skipping {file_path.name} (outputs already exist)")
            continue
        print(f"[{index}/{len(input_files)}] Processing {file_path.name}")
        processed = process_point_cloud(
            point_cloud=file_path,
            frame=frame,
            calib=calib,
            calib_file=calib_file,
            image_source=image_file,
            video_source=video_file,
            cfg=cfg,
            output_dir=output_dir,
            summary_only=args.summary_only,
        )
        results.append(processed)
        print(
            f"  input_points={processed['num_input_points']} "
            f"projected_points={processed['num_projected_points']}"
        )

    batch_summary_path = output_dir / args.summary_name
    batch_summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved batch summary: {batch_summary_path}")


if __name__ == "__main__":
    main()
