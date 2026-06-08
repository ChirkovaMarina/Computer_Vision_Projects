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


class TramRailDetector:
    def __init__(self, video_path: Path, calib_path: Path, gauge_width_m: float = 1.52):
        self.video_path = video_path
        self.gauge_width_m = gauge_width_m
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
    def _detect_rail_candidates(frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=80,
            minLineLength=80,
            maxLineGap=20,
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
            if length < 100:
                continue
            if abs(dy) < 60:
                continue
            slope = dx / (dy if dy != 0 else 1e-6)
            if abs(slope) > 0.8:
                continue
            if max(y1, y2) < height * 0.35:
                continue

            bottom = (x1, y1) if y1 > y2 else (x2, y2)
            top = (x2, y2) if y1 > y2 else (x1, y1)
            selected.append(
                {
                    "line": (x1, y1, x2, y2),
                    "bottom": bottom,
                    "top": top,
                    "slope": slope,
                    "length": length,
                }
            )
        return selected

    def _select_rail_pair(self, candidates: list[dict]):
        left_lines = []
        right_lines = []
        for item in candidates:
            item["bottom_3d"] = self._reproject_to_ground(item["bottom"])
            item["top_3d"] = self._reproject_to_ground(item["top"])
            if item["slope"] < -0.15:
                left_lines.append(item)
            elif item["slope"] > 0.15:
                right_lines.append(item)

        best_pair = None
        best_score = float("inf")
        for left in left_lines:
            for right in right_lines:
                y_diff = abs(float(left["bottom_3d"][1] - right["bottom_3d"][1]))
                if y_diff > 3.0:
                    continue

                gauge = abs(float(right["bottom_3d"][0] - left["bottom_3d"][0]))
                score = abs(gauge - self.gauge_width_m) + 0.2 * y_diff
                if score < best_score:
                    best_score = score
                    best_pair = {
                        "left": left,
                        "right": right,
                        "gauge": gauge,
                        "y_diff": y_diff,
                        "score": score,
                    }
        return best_pair

    @staticmethod
    def _draw_local_map(pair: dict | None, width: int = 420, height: int = 540) -> np.ndarray:
        canvas = np.full((height, width, 3), 25, dtype=np.uint8)
        center_x = width // 2
        origin_y = height - 30
        scale = 20.0

        cv2.line(canvas, (center_x, 20), (center_x, origin_y), (80, 80, 80), 1)
        cv2.line(canvas, (40, origin_y), (width - 40, origin_y), (80, 80, 80), 1)
        cv2.putText(canvas, "Local map", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if pair is None:
            return canvas

        for label, item, color in [
            ("L", pair["left"], (0, 255, 0)),
            ("R", pair["right"], (0, 0, 255)),
        ]:
            x, y, _ = item["bottom_3d"]
            map_x = int(center_x + x * scale)
            map_y = int(origin_y - y * scale)
            if 0 <= map_x < width and 0 <= map_y < height:
                cv2.circle(canvas, (map_x, map_y), 6, color, -1)
                cv2.putText(canvas, f"{label}: ({x:.2f}, {y:.2f})", (20, 60 if label == "L" else 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(canvas, f"Gauge: {pair['gauge']:.3f} m", (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return canvas

    def draw_result(self, frame: np.ndarray, pair: dict | None) -> np.ndarray:
        vis = frame.copy()
        if pair is not None:
            for item, color, label in [
                (pair["left"], (0, 255, 0), "L"),
                (pair["right"], (0, 0, 255), "R"),
            ]:
                x1, y1, x2, y2 = item["line"]
                bx, by = item["bottom"]
                px, py, _ = item["bottom_3d"]
                cv2.line(vis, (x1, y1), (x2, y2), color, 3)
                cv2.circle(vis, (bx, by), 5, (0, 255, 255), -1)
                cv2.putText(vis, f"{label}: ({px:.2f}, {py:.2f})", (bx + 5, by - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, f"Estimated gauge: {pair['gauge']:.3f} m", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(vis, "Rail pair not found", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        local_map = self._draw_local_map(pair, height=frame.shape[0])
        return np.hstack((vis, local_map))

    def run(self):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            candidates = self._detect_rail_candidates(frame)
            pair = self._select_rail_pair(candidates)
            vis = self.draw_result(frame, pair)
            cv2.imshow("Module_5 Task 2", vis)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = TramRailDetector(
        video_path=PROJECT_ROOT / "data" / "city" / "trm.169.007.avi",
        calib_path=PROJECT_ROOT / "data" / "city" / "leftImage.yml",
        gauge_width_m=1.52,
    )
    detector.run()
