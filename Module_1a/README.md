# Module 1a — Camera Geometry and Projection

This module contains tasks related to camera calibration usage, 3D geometry, ground-plane projection, bird's-eye-view transforms, telemetry overlays and simple path visualization.

## Programs

- `task_1.py` — projects the vehicle path onto the ground plane using camera calibration.
- `task_2.py` — projects a configurable 3D cuboid into the image.
- `task_3.py` — rotates and renders 3D objects using Euler angles.
- `task_4.py` — rotates 3D objects using quaternions.
- `task_5.py` — computes and renders a future tram path from GPS metadata.
- `task_6.py` — overlays telemetry, GPS coordinates, compass, map and logo onto video frames.
- `task_7.py` — generates a bird's-eye-view transform of the input video.
- `road_projection.py` — projects road or spline geometry into the camera image.
- `segmentation_rail.py` — segments the road area and hides everything outside the road mask.

## Related Support Code

- `srccam/calib.py` — calibration container.
- `srccam/camera.py` — 3D-to-2D projection helper.
- `srccam/point.py` — 3D point representation.
- `srccam/object3d.py` — cuboid geometry and drawing utilities.
