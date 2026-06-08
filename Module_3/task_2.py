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


class GroundPoseEstimator:
    def __init__(self, video_path: Path, calib_path: Path):
        self.video_path = video_path
        calib_dict = CalibReader(
            file_name=str(calib_path),
            param=["K", "D", "r", "t"],
        ).read()
        self.calib = Calib(calib_dict)
        self.camera = Camera(self.calib)
        self.a_inv = self._build_ground_projection_matrix()
        self.reference_rotation = self.calib.cam_to_vr @ self.calib.r
        self.reference_translation = -self.calib.cam_to_vr @ self.calib.t
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

        # For ground points z = 0, so the third column is folded into translation.
        a = np.zeros((3, 3), dtype=float)
        a[0] = [projection[0, 0], projection[0, 1], projection[0, 3]]
        a[1] = [projection[1, 0], projection[1, 1], projection[1, 3]]
        a[2] = [projection[2, 0], projection[2, 1], projection[2, 3]]
        return np.linalg.inv(a)

    def _road_mask(self, frame_shape: tuple[int, int, int]) -> np.ndarray:
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, self.road_polygon, 255)
        return mask

    def _detect_harris_points(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=150,
            qualityLevel=0.01,
            minDistance=8,
            mask=mask,
            useHarrisDetector=True,
            k=0.04,
        )

    def _reproject_to_ground(self, points_2d: np.ndarray) -> np.ndarray:
        object_points = []
        for x, y in points_2d:
            uv1 = np.array([[x], [y], [1.0]], dtype=float)
            xyz = self.a_inv @ uv1
            xyz = xyz[:, 0] / xyz[2, 0]
            object_points.append([xyz[0], xyz[1], 0.0])
        return np.asarray(object_points, dtype=np.float32)

    @staticmethod
    def _rotation_to_euler_degrees(rotation: np.ndarray) -> np.ndarray:
        angles = cv2.RQDecomp3x3(rotation)[0]
        return np.asarray(angles, dtype=float)

    def _estimate_pose(self, prev_gray: np.ndarray, frame_gray: np.ndarray, frame_shape: tuple[int, int, int]):
        mask = self._road_mask(frame_shape)
        points = self._detect_harris_points(prev_gray, mask)
        if points is None or len(points) < 8:
            return None

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, points, None)
        if next_points is None or status is None:
            return None

        prev_points = points.reshape(-1, 2)
        next_points = next_points.reshape(-1, 2)
        status = status.reshape(-1).astype(bool)

        prev_points = prev_points[status]
        next_points = next_points[status]
        if len(prev_points) < 8:
            return None

        object_points = self._reproject_to_ground(prev_points)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            next_points,
            self.calib.K,
            self.calib.D,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok or inliers is None or len(inliers) < 6:
            return None

        inlier_idx = inliers.reshape(-1)
        rotation, _ = cv2.Rodrigues(rvec)
        return {
            "prev_points": prev_points,
            "next_points": next_points,
            "inlier_idx": inlier_idx,
            "rotation": rotation,
            "translation": tvec,
        }

    def _draw_result(self, frame: np.ndarray, result: dict | None) -> np.ndarray:
        output = frame.copy()
        cv2.polylines(output, [self.road_polygon], isClosed=True, color=(0, 255, 255), thickness=2)

        if result is None:
            cv2.putText(output, "Pose estimation failed", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return output

        prev_points = result["prev_points"]
        next_points = result["next_points"]
        inlier_idx = set(result["inlier_idx"].tolist())

        for idx, (prev_pt, next_pt) in enumerate(zip(prev_points, next_points)):
            prev_xy = tuple(np.round(prev_pt).astype(int))
            next_xy = tuple(np.round(next_pt).astype(int))
            color = (255, 0, 0) if idx in inlier_idx else (80, 80, 80)
            cv2.circle(output, prev_xy, 2, (0, 255, 0), -1)
            cv2.circle(output, next_xy, 2, (0, 0, 255), -1)
            cv2.line(output, prev_xy, next_xy, color, 1)

        estimated_angles = self._rotation_to_euler_degrees(result["rotation"])
        reference_angles = self._rotation_to_euler_degrees(self.reference_rotation)
        estimated_t = result["translation"].reshape(-1)
        reference_t = self.reference_translation.reshape(-1)

        info_lines = [
            f"Inliers: {len(inlier_idx)} / {len(prev_points)}",
            "Angles ref/est (deg):",
            f"roll  {reference_angles[0]:7.3f} / {estimated_angles[0]:7.3f}",
            f"pitch {reference_angles[1]:7.3f} / {estimated_angles[1]:7.3f}",
            f"yaw   {reference_angles[2]:7.3f} / {estimated_angles[2]:7.3f}",
            "Position ref/est:",
            f"x {reference_t[0]:7.3f} / {estimated_t[0]:7.3f}",
            f"y {reference_t[1]:7.3f} / {estimated_t[1]:7.3f}",
            f"z {reference_t[2]:7.3f} / {estimated_t[2]:7.3f}",
        ]

        for index, line in enumerate(info_lines):
            cv2.putText(
                output,
                line,
                (20, 30 + index * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return output

    def run(self):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")

        ok, prev_frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read first frame: {self.video_path}")

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = self._estimate_pose(prev_gray, frame_gray, frame.shape)
            output = self._draw_result(frame, result)
            cv2.imshow("Task 2", output)

            prev_gray = frame_gray
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    estimator = GroundPoseEstimator(
        video_path=PROJECT_ROOT / "data" / "city" / "trm.169.007.avi",
        calib_path=PROJECT_ROOT / "data" / "city" / "leftImage.yml",
    )
    estimator.run()
