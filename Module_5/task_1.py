from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRCCAM_ROOT = PROJECT_ROOT / "srccam"
if str(SRCCAM_ROOT) not in sys.path:
    sys.path.insert(0, str(SRCCAM_ROOT))

from srccam.calib import Calib
from srccam.load_calib import CalibReader


class VerticalLineMapper:
    def __init__(self, video_path: Path, calib_path: Path):
        self.video_path = video_path
        calib_dict = CalibReader(
            file_name=str(calib_path),
            param=["K", "D", "r", "t"],
        ).read()
        self.calib = Calib(calib_dict)
        self.a_inv = self._build_ground_projection_matrix()

    def _build_ground_projection_matrix(self) -> np.ndarray:
        rotation = self.calib.cam_to_vr @ self.calib.r
        affine = np.concatenate((rotation, -rotation @ self.calib.t), axis=1)
        projection = self.calib.K @ affine
        a = np.zeros((3, 3), dtype=float)
        a[0] = [projection[0, 0], projection[0, 1], projection[0, 3]]
        a[1] = [projection[1, 0], projection[1, 1], projection[1, 3]]
        a[2] = [projection[2, 0], projection[2, 1], projection[2, 3]]
        return np.linalg.inv(a)

    def _reproject_to_ground(self, point_2d: tuple[int, int]) -> np.ndarray:
        uv1 = np.array([[point_2d[0]], [point_2d[1]], [1.0]], dtype=float)
        xyz = self.a_inv @ uv1
        xyz = xyz[:, 0] / xyz[2, 0]
        return np.asarray([xyz[0], xyz[1], 0.0], dtype=np.float32)

    @staticmethod
    def _detect_vertical_lines(frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=60,
            minLineLength=40,
            maxLineGap=10,
        )
        if lines is None:
            return []

        selected = []
        height = frame.shape[0]
        for line in lines[:, 0]:
            x1, y1, x2, y2 = map(int, line)
            dx = x2 - x1
            dy = y2 - y1
            length = float(np.hypot(dx, dy))
            if length < 50:
                continue
            if abs(dy) < 40:
                continue
            if abs(dx) > 0.25 * abs(dy):
                continue
            bottom_y = max(y1, y2)
            if bottom_y < height * 0.35:
                continue
            selected.append((x1, y1, x2, y2, length))
        return selected

    def analyze_frame(self, frame: np.ndarray):
        lines = self._detect_vertical_lines(frame)
        mapped_points = []
        for x1, y1, x2, y2, length in lines:
            bottom_point = (x1, y1) if y1 > y2 else (x2, y2)
            point_3d = self._reproject_to_ground(bottom_point)
            x, y, _ = point_3d

            # Reject points with unrealistic local-map coordinates.
            if not (-10.0 <= x <= 10.0 and 0.0 <= y <= 40.0):
                continue

            mapped_points.append(
                {
                    "line": (x1, y1, x2, y2),
                    "bottom_point": bottom_point,
                    "point_3d": point_3d,
                    "length": length,
                }
            )
        return mapped_points

    @staticmethod
    def _draw_local_map(points: list[dict], width: int = 420, height: int = 540) -> np.ndarray:
        canvas = np.full((height, width, 3), 25, dtype=np.uint8)
        center_x = width // 2
        origin_y = height - 30
        scale = 10.0

        cv2.line(canvas, (center_x, 20), (center_x, origin_y), (80, 80, 80), 1)
        cv2.line(canvas, (40, origin_y), (width - 40, origin_y), (80, 80, 80), 1)
        cv2.putText(canvas, "Local map", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, "x", (width - 25, origin_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.putText(canvas, "y", (center_x + 8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        for idx, item in enumerate(points, start=1):
            x, y, _ = item["point_3d"]
            map_x = int(center_x + x * scale)
            map_y = int(origin_y - y * scale)
            if 0 <= map_x < width and 0 <= map_y < height:
                cv2.circle(canvas, (map_x, map_y), 5, (0, 255, 255), -1)
                cv2.putText(
                    canvas,
                    f"{idx}: ({x:.1f}, {y:.1f})",
                    (20, 60 + idx * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )
        return canvas

    def draw_result(self, frame: np.ndarray, points: list[dict]) -> np.ndarray:
        frame_vis = frame.copy()
        for idx, item in enumerate(points, start=1):
            x1, y1, x2, y2 = item["line"]
            bx, by = item["bottom_point"]
            px, py, _ = item["point_3d"]
            cv2.line(frame_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(frame_vis, (bx, by), 4, (0, 255, 255), -1)
            cv2.putText(
                frame_vis,
                f"{idx}: ({px:.1f}, {py:.1f})",
                (bx + 5, by - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame_vis,
            f"Vertical lines on ground: {len(points)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        local_map = self._draw_local_map(points, height=frame.shape[0])
        return np.hstack((frame_vis, local_map))

    def run(self):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            points = self.analyze_frame(frame)
            vis = self.draw_result(frame, points)
            cv2.imshow("Module_5 Task 1", vis)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    mapper = VerticalLineMapper(
        video_path=PROJECT_ROOT / "data" / "city" / "trm.169.007.avi",
        calib_path=PROJECT_ROOT / "data" / "city" / "leftImage.yml",
    )
    mapper.run()
