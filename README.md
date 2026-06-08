# Computer Vision in Autonomous Driving Tasks

This repository contains course materials and practical scripts for classical computer vision tasks used in autonomous driving: image preprocessing, camera calibration, geometry, feature detection, Hough-based analysis, and LiDAR processing.

## Requirements

- Python 3.9+
- OpenCV
- NumPy

Some modules additionally use:

- `matplotlib`
- `PyYAML`
- `laspy[lazrs]` for LiDAR `.laz` files
- `torch` for the PointNet-based LiDAR experiments

## Installation

With `Makefile`:

```bash
make fullinstall
```

Or manually:

```bash
pip install .
```

The repository also contains scripts that expect local data under `data/`.

## Repository Structure

### `Module_1` — Basic Image Operations

Introductory OpenCV examples for image loading, color conversion, resizing, drawing and basic pixel manipulation.

- `1_imread.py` — read and display an image.
- `2_cvtcolor.py` — convert between color spaces.
- `3_resize.py` — resize an image.
- `4_draw.py` — draw lines, shapes and text.
- `5_pyr_up_down.py` — pyramid up/down scaling.
- `6_cv_merge.py` — split and merge image channels.
- `7_crop_flip.py` — crop and flip an image.
- `8_copyMakeBorder.py` — add image borders.
- `9_arithmetic_operation.py` — arithmetic image operations.
- `10_bitwise_operation.py` — bitwise operations and masks.
- `graph_draw.py` — draw a mathematical graph into an image.
- `newtask__paint.py` — interactive mouse drawing demo.

### `Module_1a` — Camera Geometry and Projection

Tasks related to camera calibration usage, 3D-to-2D projection, bird's-eye view, path rendering and telemetry overlays.

- `task_1.py` — draw the projected vehicle path on the ground plane.
- `task_2.py` — project a 3D cuboid of configurable size into the image.
- `task_3.py` — rotate and render 3D objects using Euler angles.
- `task_4.py` — rotate 3D objects using quaternions.
- `task_5.py` — estimate and render a future tram path from GPS metadata.
- `task_6.py` — overlay telemetry, coordinates, compass and logo on the frame.
- `task_7.py` — create a bird's-eye-view transformation.
- `road_projection.py` — project road/spline geometry into the image.
- `segmentation_rail.py` — highlight the road area and mask everything outside it.

### `Module_1b` — Camera Calibration

Examples of camera calibration and pose visualization using checkerboards and circle grids.

- `calibration.py` — standard chessboard camera calibration and undistortion.
- `calibration_fisheye.py` — fisheye calibration and reprojection visualization.
- `camera_motion.py` — visualize camera motion relative to the calibration pattern.
- `Circles_calib.py` — calibration using an asymmetric circle grid.

### `Module_2` — Image Preprocessing and Visibility Enhancement

Classical image filtering and visibility-improvement tasks for automotive video.

- `Filters.py` — blur and convolution filter examples.
- `ColorSpaces.py` — color space examples and channel operations.
- `Translations.py` — affine translation examples.
- `Rotations.py` — image rotation examples.
- `sharpened.py` — sharpening filter for video frames.
- `Sun_blicks.py` — reduce sun glare on glass.
- `detection_flares.py` — detect glare regions using HLS.
- `reduce_glare.py` — polynomial and gamma-based glare reduction.
- `task_1.py` — reduce headlight glare using a distance-mask approach.
- `task_2.py` — reduce overexposure caused by a snow storm.
- `task_3.py` — suppress windshield wiper and raindrop effects with frame differencing.
- `task_4.py` — reduce glare in the upper camera area under direct sunlight.
- `task_5.py` — reduce sun glare on wet asphalt.
- `task_5(right).py` — reduce glare from oncoming headlights.
- `task_7.py` — another direct-sun overexposure reduction task.

### `Module_2a` — Morphological Operations

Examples of binary morphology and segmentation.

- `1_threshold.py` — thresholding demo.
- `2_morphologyEx.py` — advanced morphology operations.
- `3_hitmiss.py` — hit-or-miss transform example.
- `4_erod_and_dilate.py` — erosion and dilation examples.
- `Morphological_Operations_segmentation.py` — segmentation pipeline based on morphology.

### `Module_3` — Corner Detectors and Geometric Reasoning

Tasks based on Harris/FAST features and camera-ground geometry.

- `task_1.py` — project Harris points from one frame to the next using known forward motion.
- `task_2.py` — estimate camera placement from 3D points on a ground region.
- `task_3.py` — compare Harris and FAST points, overlaps and percentages.
- `task_4.py` — compute 3D coordinates of Harris points on the road and test coplanarity.
- `task_5.py` — randomly sample Harris points until a planar road subset is found.
- `fast_detector.py` — standalone FAST detector demo.

### `Module_4a` — Keypoints, Matching and Motion Analysis

Feature-based tasks with SIFT/ORB and geometric filtering.

- `task1.py` — SIFT matching between frames, ground filtering and 3D motion estimation.
- `task2.py` — SIFT matching with homography estimation and RANSAC outlier rejection.
- `task3.py` — determine whether the observed object is moving based on keypoint displacement.
- `task5.py` — detect moving objects while the tram itself is stationary.

### `Module_4b` — Advanced Matching / Learned Features

References and notes for advanced feature pipelines.

- `readme.txt` — notes and example commands for SuperPoint and SuperGlue experiments.

### `Module_5` — Hough Transform and Projective Geometry

Tasks based on line/circle detection, vanishing points and rail geometry.

- `task_1.py` — detect vertical lines and map their ground contact points to local coordinates.
- `task_2.py` — detect tram rails using the known track gauge.
- `task_3.py` — estimate the horizon line from rail-line intersections and a vanishing point.
- `task_4.py` — estimate precise camera pitch from rail projections.
- `task_5.py` — detect traffic lights using Hough circles.
- `simple.py` — simple Hough transform implementation and visualization.
- `vanishing_point.py` — vanishing point estimation from line intersections.

### `Module_5c` — Graph-Based Segmentation

Small graph-based segmentation examples.

- `graph.py` — graph-based segmentation helper/demo.

### `srccam` — Shared Geometry and Sensor Utilities

Reusable support package used by several modules.

- `calib.py` — calibration parameter container.
- `camera.py` — 3D-to-2D projection helper.
- `load_calib.py` — calibration file reader.
- `point.py` — 3D point representation.
- `object3d.py` — cuboid geometry and drawing.
- `season_reader.py` — base class for processing season/video metadata streams.
- `sense_data.py` — telemetry and GPS/IMU helper.

### `data/LIDAR` — LiDAR Experiments

Scripts for LiDAR visualization, ground separation, clustering, multiple point cloud representations, PointNet experiments and simple SLAM/mapping.

- `visualize_lidar.py` — visualize a `.laz` scan in 3D and BEV.
- `lidar_ground.py` — separate ground and non-ground points with RANSAC.
- `lidar_clusters.py` — cluster above-ground obstacle points.
- `lidar_point_based.py` — point-based local neighborhood features.
- `lidar_voxelization.py` — voxelize the cloud and visualize voxel centroids.
- `lidar_bev.py` — build BEV density and height maps.
- `lidar_point_pillars.py` — simple PointPillars-style encoding.
- `lidar_frontal_view.py` — generate a frontal/range image representation.
- `lidar_sparse_conv.py` — prepare sparse-convolution-ready coordinates and features.
- `lidar_slam_mapping.py` — build a simple multi-frame LiDAR map with sequential ICP alignment.
- `prepare_pointnet_dataset.py` — generate a pseudo-labeled dataset for ground/non-ground learning.
- `pointnet_ground.py` — train a small PointNet for ground segmentation.
- `predict_pointnet_ground.py` — run PointNet inference and compare it with RANSAC.
- `train_minkowski_ground.py` — Linux/CUDA-oriented sparse model example for MinkowskiEngine.

### `homework`

Assignment PDFs for the course:

- `homework-1.pdf`
- `homework-2.pdf`
- `homework-3.pdf`

## Notes

- Some scripts are pure demos and expect manual interaction through `cv2.imshow(...)`.
- Many tasks depend on local data under `data/`.
- LiDAR sparse-convolution backends such as `MinkowskiEngine` or `spconv` are not expected to run on macOS Apple Silicon without a separate Linux/CUDA environment.
