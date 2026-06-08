from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRCCAM_ROOT = PROJECT_ROOT / "srccam"
if str(SRCCAM_ROOT) not in sys.path:
    sys.path.insert(0, str(SRCCAM_ROOT))

from srccam.calib import Calib
from srccam.camera import Camera
from srccam.load_calib import CalibReader
from srccam.point import Point3d as Point


class RandomRoadPointSearch:
    def __init__(self, video_path: Path, calib_path: Path, sample_size: int = 6, max_attempts: int = 500):
        self.video_path = video_path
        self.sample_size = sample_size
        self.max_attempts = max_attempts
        calib_dict = CalibReader(
            file_name=str(calib_path),
            param=["K", "D", "r", "t"],
        ).read()
        self.calib = Calib(calib_dict)
        self.camera = Camera(self.calib)
        self.a_inv = self._build_ground_projection_matrix()
        self.road_polygon = np.array(
            [
                self.camera.project_point_3d_to_2d(Point((-1.8, 6.0, 0.0))),
                self.camera.project_point_3d_to_2d(Point((1.8, 6.0, 0.0))),
                self.camera.project_point_3d_to_2d(Point((2.5, 30.0, 0.0))),
                self.camera.project_point_3d_to_2d(Point((-2.5, 30.0, 0.0))),
            ],
            dtype=np.int32,
        )

    def _build_ground_projection_matrix(self) -> np.ndarray:
        rotation = self.calib.cam_to_vr @ self.calib.r
        affine = np.concatenate((rotation, -rotation @ self.calib.t), axis=1)
        projection = self.calib.K @ affine
        a = np.zeros((3, 3), dtype=float)
        a[0] = [projection[0, 0], projection[0, 1], projection[0, 3]]
        a[1] = [projection[1, 0], projection[1, 1], projection[1, 3]]
        a[2] = [projection[2, 0], projection[2, 1], projection[2, 3]]
        return np.linalg.inv(a)

    def _load_frame(self) -> np.ndarray:
        cap = cv2.VideoCapture(str(self.video_path))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise FileNotFoundError(f"Cannot read first frame: {self.video_path}")
        return frame

    @staticmethod
    def _detect_harris_points(gray: np.ndarray) -> np.ndarray:
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=250,
            qualityLevel=0.01,
            minDistance=8,
            useHarrisDetector=True,
            k=0.04,
        )
        if points is None:
            return np.empty((0, 2), dtype=np.float32)
        return points.reshape(-1, 2)

    def _is_on_road(self, point_2d: np.ndarray) -> bool:
        return cv2.pointPolygonTest(self.road_polygon, (float(point_2d[0]), float(point_2d[1])), False) >= 0

    def _reproject_to_ground(self, points_2d: np.ndarray) -> np.ndarray:
        object_points = []
        for x, y in points_2d:
            uv1 = np.array([[x], [y], [1.0]], dtype=float)
            xyz = self.a_inv @ uv1
            xyz = xyz[:, 0] / xyz[2, 0]
            object_points.append([xyz[0], xyz[1], 0.0])
        return np.asarray(object_points, dtype=np.float32)

    @staticmethod
    def _fit_plane(points_3d: np.ndarray) -> tuple[np.ndarray, float, float]:
        centroid = points_3d.mean(axis=0)
        centered = points_3d - centroid
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        normal = vt[-1]
        normal = normal / np.linalg.norm(normal)
        distances = np.abs(centered @ normal)
        return normal, float(distances.max()), float(distances.mean())

    def analyze(self) -> tuple[np.ndarray, dict]:
        frame = self._load_frame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        harris_points = self._detect_harris_points(gray)
        if len(harris_points) < self.sample_size:
            raise RuntimeError("Not enough Harris points on the frame")

        rng = np.random.default_rng(42)
        shuffled_indices = rng.permutation(len(harris_points))
        selected_points = []
        attempts = 0
        for idx in shuffled_indices:
            attempts += 1
            point = harris_points[idx]
            if self._is_on_road(point):
                selected_points.append(point)
                if len(selected_points) == self.sample_size:
                    break
            if attempts >= self.max_attempts:
                break

        if len(selected_points) < self.sample_size:
            raise RuntimeError("No random sample of Harris points on the road was found")

        sample_2d = np.asarray(selected_points, dtype=np.float32)
        sample_3d = self._reproject_to_ground(sample_2d)
        normal, max_distance, mean_distance = self._fit_plane(sample_3d)
        result = {
            "attempt": attempts,
            "all_harris_points": harris_points,
            "points_2d": sample_2d,
            "points_3d": sample_3d,
            "plane_normal": normal,
            "max_distance": max_distance,
            "mean_distance": mean_distance,
            "is_planar": max_distance < 1e-4,
        }
        return frame, result

    def draw_result(self, frame: np.ndarray, result: dict) -> np.ndarray:
        output = frame.copy()
        cv2.polylines(output, [self.road_polygon], isClosed=True, color=(0, 255, 255), thickness=2)

        for point in result["all_harris_points"]:
            xy = tuple(np.round(point).astype(int))
            cv2.circle(output, xy, 2, (120, 120, 120), -1)

        for point in result["points_2d"]:
            xy = tuple(np.round(point).astype(int))
            cv2.circle(output, xy, 5, (0, 0, 255), -1)

        normal = result["plane_normal"]
        lines = [
            f"Attempts: {result['attempt']}",
            f"Sample size: {len(result['points_2d'])}",
            f"Plane normal: [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]",
            f"Max dist to plane: {result['max_distance']:.6f}",
            f"Mean dist to plane: {result['mean_distance']:.6f}",
            f"Road sample found: yes",
            f"One plane: {'yes' if result['is_planar'] else 'no'}",
        ]
        for index, line in enumerate(lines):
            cv2.putText(
                output,
                line,
                (20, 30 + index * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return output


if __name__ == "__main__":
    analyzer = RandomRoadPointSearch(
        video_path=PROJECT_ROOT / "data" / "city" / "trm.169.007.avi",
        calib_path=PROJECT_ROOT / "data" / "city" / "leftImage.yml",
        sample_size=6,
        max_attempts=500,
    )
    frame, result = analyzer.analyze()
    print(f"Attempt: {result['attempt']}")
    print(f"Sample size: {len(result['points_2d'])}")
    print("Selected 2D points:")
    print(result["points_2d"])
    print("Selected 3D points:")
    print(result["points_3d"])
    print(f"Plane normal: {result['plane_normal']}")
    print(f"Max distance to plane: {result['max_distance']:.8f}")
    print(f"Mean distance to plane: {result['mean_distance']:.8f}")
    print(f"All points in one plane: {'yes' if result['is_planar'] else 'no'}")

    output = analyzer.draw_result(frame, result)
    cv2.imshow("Task 5", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
