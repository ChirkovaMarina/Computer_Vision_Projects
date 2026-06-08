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


class CameraPitchEstimator:
    def __init__(self, video_path: Path, calib_path: Path, gauge_width_m: float = 1.52):
        self.video_path = video_path
        self.gauge_width_m = gauge_width_m
        self.calib_dict = CalibReader(
            file_name=str(calib_path),
            param=["K", "D", "r", "t"],
        ).read()
        self.calib = Calib(self.calib_dict)
        self.base_angles = np.asarray(self.calib_dict["r"], dtype=float).reshape(-1)
        self.K = np.asarray(self.calib_dict["K"], dtype=float)
        self.t = np.asarray(self.calib_dict["t"], dtype=float).reshape(3, 1)
        self.cam_to_vr = self.calib.cam_to_vr

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
            if length < 100 or abs(dy) < 60:
                continue
            slope = dx / (dy if dy != 0 else 1e-6)
            if abs(slope) > 0.8 or max(y1, y2) < height * 0.35:
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

    def _build_ground_projection_matrix(self, rotation: np.ndarray) -> np.ndarray:
        affine = np.concatenate((rotation, -rotation @ self.t), axis=1)
        projection = self.K @ affine
        a = np.zeros((3, 3), dtype=float)
        a[0] = [projection[0, 0], projection[0, 1], projection[0, 3]]
        a[1] = [projection[1, 0], projection[1, 1], projection[1, 3]]
        a[2] = [projection[2, 0], projection[2, 1], projection[2, 3]]
        return np.linalg.inv(a)

    def _reproject_to_ground(self, point_2d: tuple[int, int]) -> np.ndarray:
        rotation = self.cam_to_vr @ self.calib.r
        a_inv = self._build_ground_projection_matrix(rotation)
        uv1 = np.array([[point_2d[0]], [point_2d[1]], [1.0]], dtype=float)
        xyz = a_inv @ uv1
        xyz = xyz[:, 0] / xyz[2, 0]
        return np.asarray([xyz[0], xyz[1], 0.0], dtype=np.float32)

    def _select_rail_pair(self, candidates: list[dict]):
        left_lines = []
        right_lines = []
        for item in candidates:
            item["bottom_3d"] = self._reproject_to_ground(item["bottom"])
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
                    best_pair = {"left": left, "right": right, "gauge": gauge}
        return best_pair

    @staticmethod
    def _line_from_points(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        a = p1[1] - p2[1]
        b = p2[0] - p1[0]
        c = p1[0] * p2[1] - p2[0] * p1[1]
        norm = np.hypot(a, b)
        return np.array([a / norm, b / norm, c / norm], dtype=float)

    @staticmethod
    def _point_line_distance(line: np.ndarray, point: np.ndarray) -> float:
        return abs(line[0] * point[0] + line[1] * point[1] + line[2])

    def _project_point(self, point_3d: tuple[float, float, float], pitch: float) -> np.ndarray:
        angles = self.base_angles.copy()
        angles[1] = pitch
        rotation = Calib.rotation_matrix_from(angles).T
        point = np.asarray(point_3d, dtype=float).reshape(3, 1)
        rotated = rotation @ point - self.t
        vr = self.cam_to_vr @ rotated
        res = self.K @ vr
        x, y, w = res.reshape(-1)
        return np.asarray([x / w, y / w], dtype=float)

    def _projection_error(self, pair: dict, pitch: float):
        near_y = min(float(pair["left"]["bottom_3d"][1]), float(pair["right"]["bottom_3d"][1]))
        far_y = max(near_y + 18.0, 25.0)

        model = {
            "left_near": self._project_point((-self.gauge_width_m / 2, near_y, 0.0), pitch),
            "left_far": self._project_point((-self.gauge_width_m / 2, far_y, 0.0), pitch),
            "right_near": self._project_point((self.gauge_width_m / 2, near_y, 0.0), pitch),
            "right_far": self._project_point((self.gauge_width_m / 2, far_y, 0.0), pitch),
        }

        observed_left = np.asarray([pair["left"]["bottom"], pair["left"]["top"]], dtype=float)
        observed_right = np.asarray([pair["right"]["bottom"], pair["right"]["top"]], dtype=float)

        model_left_line = self._line_from_points(model["left_near"], model["left_far"])
        model_right_line = self._line_from_points(model["right_near"], model["right_far"])
        observed_left_line = self._line_from_points(observed_left[0], observed_left[1])
        observed_right_line = self._line_from_points(observed_right[0], observed_right[1])

        score = 0.0
        for point in observed_left:
            score += self._point_line_distance(model_left_line, point)
        for point in observed_right:
            score += self._point_line_distance(model_right_line, point)
        for point in [model["left_near"], model["left_far"]]:
            score += self._point_line_distance(observed_left_line, point)
        for point in [model["right_near"], model["right_far"]]:
            score += self._point_line_distance(observed_right_line, point)
        return score, model

    def _estimate_pitch(self, pair: dict):
        base_pitch = float(self.base_angles[1])
        best_pitch = base_pitch
        best_score = float("inf")
        best_model = None

        # Coarse-to-fine search around the calibrated pitch.
        search_ranges = [(0.10, 81), (0.02, 81), (0.005, 81)]
        current_center = base_pitch
        for radius, steps in search_ranges:
            candidates = np.linspace(current_center - radius, current_center + radius, steps)
            for pitch in candidates:
                score, model = self._projection_error(pair, float(pitch))
                if score < best_score:
                    best_score = score
                    best_pitch = float(pitch)
                    best_model = model
            current_center = best_pitch

        return best_pitch, best_score, best_model

    def draw_result(self, frame: np.ndarray, pair: dict | None, estimated_pitch: float | None, score: float | None, model: dict | None):
        vis = frame.copy()
        if pair is None:
            cv2.putText(vis, "Rail pair not found", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return vis

        for item, color in [(pair["left"], (0, 255, 0)), (pair["right"], (0, 0, 255))]:
            x1, y1, x2, y2 = item["line"]
            cv2.line(vis, (x1, y1), (x2, y2), color, 3)

        if model is not None:
            left_near = tuple(np.round(model["left_near"]).astype(int))
            left_far = tuple(np.round(model["left_far"]).astype(int))
            right_near = tuple(np.round(model["right_near"]).astype(int))
            right_far = tuple(np.round(model["right_far"]).astype(int))
            cv2.line(vis, left_near, left_far, (255, 255, 0), 2)
            cv2.line(vis, right_near, right_far, (0, 255, 255), 2)

        base_pitch_deg = float(np.degrees(self.base_angles[1]))
        if estimated_pitch is not None:
            est_pitch_deg = float(np.degrees(estimated_pitch))
            lines = [
                f"Base pitch: {base_pitch_deg:.3f} deg",
                f"Estimated pitch: {est_pitch_deg:.3f} deg",
                f"Delta pitch: {est_pitch_deg - base_pitch_deg:.3f} deg",
                f"Projection error: {score:.3f}",
            ]
        else:
            lines = [f"Base pitch: {base_pitch_deg:.3f} deg"]

        for idx, line in enumerate(lines):
            cv2.putText(vis, line, (20, 30 + idx * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return vis

    def run(self):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")

        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            pair = self._select_rail_pair(self._detect_rail_candidates(frame))
            estimated_pitch = None
            score = None
            model = None
            if pair is not None:
                estimated_pitch, score, model = self._estimate_pitch(pair)

            vis = self.draw_result(frame, pair, estimated_pitch, score, model)
            if frame_index == 0:
                output_path = PROJECT_ROOT / "Module_5" / "task_4_result.png"
                cv2.imwrite(str(output_path), vis)
                print(f"Saved preview: {output_path}")

            base_pitch_deg = float(np.degrees(self.base_angles[1]))
            if estimated_pitch is None:
                print(f"Frame {frame_index}: rail pair not found")
            else:
                estimated_pitch_deg = float(np.degrees(estimated_pitch))
                print(
                    f"Frame {frame_index}: "
                    f"base_pitch={base_pitch_deg:.4f} deg, "
                    f"estimated_pitch={estimated_pitch_deg:.4f} deg, "
                    f"delta={estimated_pitch_deg - base_pitch_deg:.4f} deg, "
                    f"error={score:.4f}"
                )

            cv2.imshow("Module_5 Task 4", vis)
            frame_index += 1
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    estimator = CameraPitchEstimator(
        video_path=PROJECT_ROOT / "data" / "city" / "trm.169.007.avi",
        calib_path=PROJECT_ROOT / "data" / "city" / "leftImage.yml",
        gauge_width_m=1.52,
    )
    estimator.run()
