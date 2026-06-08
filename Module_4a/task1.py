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


class SIFTGroundMotionEstimator:
    def __init__(self, video_path: Path, calib_path: Path):
        self.video_path = video_path
        calib_dict = CalibReader(
            file_name=str(calib_path),
            param=["K", "D", "r", "t"],
        ).read()
        self.calib = Calib(calib_dict)
        self.camera = Camera(self.calib)
        self.a_inv = self._build_ground_projection_matrix()
        self.sift = cv2.SIFT_create(nfeatures=800)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
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

    def _point_on_road(self, point_2d: tuple[float, float]) -> bool:
        return cv2.pointPolygonTest(self.road_polygon, point_2d, False) >= 0

    def _reproject_to_ground(self, point_2d: tuple[float, float]) -> np.ndarray:
        uv1 = np.array([[point_2d[0]], [point_2d[1]], [1.0]], dtype=float)
        xyz = self.a_inv @ uv1
        xyz = xyz[:, 0] / xyz[2, 0]
        return np.asarray([xyz[0], xyz[1], 0.0], dtype=np.float32)

    def _match_ground_points(self, frame1: np.ndarray, frame2: np.ndarray):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.sift.detectAndCompute(gray1, None)
        kp2, des2 = self.sift.detectAndCompute(gray2, None)
        if des1 is None or des2 is None:
            return kp1, kp2, [], np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        knn_matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        points3d_1 = []
        points3d_2 = []

        for pair in knn_matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance >= 0.75 * second.distance:
                continue

            point1 = kp1[first.queryIdx].pt
            point2 = kp2[first.trainIdx].pt
            if not self._point_on_road(point1) or not self._point_on_road(point2):
                continue

            good_matches.append(first)
            points3d_1.append(self._reproject_to_ground(point1))
            points3d_2.append(self._reproject_to_ground(point2))

        return (
            kp1,
            kp2,
            good_matches,
            np.asarray(points3d_1, dtype=np.float32),
            np.asarray(points3d_2, dtype=np.float32),
        )

    @staticmethod
    def _estimate_translation(points3d_1: np.ndarray, points3d_2: np.ndarray) -> np.ndarray:
        if len(points3d_1) == 0:
            return np.zeros(3, dtype=np.float32)
        displacements = points3d_2 - points3d_1
        return np.median(displacements, axis=0)

    def _draw_result(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        kp1,
        kp2,
        matches,
        translation: np.ndarray,
    ) -> np.ndarray:
        left = frame1.copy()
        right = frame2.copy()
        cv2.polylines(left, [self.road_polygon], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.polylines(right, [self.road_polygon], isClosed=True, color=(0, 255, 255), thickness=2)

        match_vis = cv2.drawMatches(
            left,
            kp1,
            right,
            kp2,
            matches[:40],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        lines = [
            f"Ground matches: {len(matches)}",
            f"Estimated dx: {translation[0]:.3f} m",
            f"Estimated dy: {translation[1]:.3f} m",
            f"Estimated dz: {translation[2]:.3f} m",
        ]
        for idx, line in enumerate(lines):
            cv2.putText(
                match_vis,
                line,
                (20, 30 + idx * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return match_vis

    def run(self):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")

        ok, prev_frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read first frame: {self.video_path}")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            kp1, kp2, matches, points3d_1, points3d_2 = self._match_ground_points(prev_frame, frame)
            translation = self._estimate_translation(points3d_1, points3d_2)
            vis = self._draw_result(prev_frame, frame, kp1, kp2, matches, translation)
            cv2.imshow("Module_4a Task 1", vis)

            prev_frame = frame
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    estimator = SIFTGroundMotionEstimator(
        video_path=PROJECT_ROOT / "data" / "city" / "trm.169.007.avi",
        calib_path=PROJECT_ROOT / "data" / "city" / "leftImage.yml",
    )
    estimator.run()
