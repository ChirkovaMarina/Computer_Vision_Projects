from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TrafficLightDetector:
    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.color_specs = {
            "RED": {
                "ranges": [
                    (np.array([0, 100, 120]), np.array([10, 255, 255])),
                    (np.array([160, 100, 120]), np.array([180, 255, 255])),
                ],
                "draw_color": (0, 0, 255),
                "fill_threshold": 0.28,
            },
            "GREEN": {
                "ranges": [
                    (np.array([35, 60, 80]), np.array([95, 255, 255])),
                ],
                "draw_color": (0, 255, 0),
                "fill_threshold": 0.50,
            },
            "YELLOW": {
                "ranges": [
                    (np.array([15, 120, 120]), np.array([40, 255, 255])),
                ],
                "draw_color": (0, 255, 255),
                "fill_threshold": 0.30,
            },
        }

    @staticmethod
    def _build_mask(hsv: np.ndarray, ranges: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    @staticmethod
    def _circle_fill_ratio(mask: np.ndarray, x: int, y: int, radius: int) -> float:
        if radius <= 0:
            return 0.0
        yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
        circle = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
        if not np.any(circle):
            return 0.0
        return float(mask[circle].mean() / 255.0)

    @staticmethod
    def _deduplicate(detections: list[dict]) -> list[dict]:
        result = []
        for det in sorted(detections, key=lambda item: item["score"], reverse=True):
            keep = True
            for kept in result:
                distance = np.hypot(det["x"] - kept["x"], det["y"] - kept["y"])
                if det["label"] == kept["label"] and distance < max(det["r"], kept["r"], 8):
                    keep = False
                    break
            if keep:
                result.append(det)
        return result

    def detect(self, frame: np.ndarray) -> list[dict]:
        blurred = cv2.medianBlur(frame, 5)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Traffic lights are expected in the upper part of the image.
        roi_limit = int(frame.shape[0] * 0.45)
        roi_left = int(frame.shape[1] * 0.60)
        detections = []
        for label, spec in self.color_specs.items():
            mask = self._build_mask(hsv, spec["ranges"])
            mask[roi_limit:, :] = 0
            mask[:, :roi_left] = 0
            circles = cv2.HoughCircles(
                mask,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=20,
                param1=80,
                param2=8,
                minRadius=3,
                maxRadius=14,
            )
            if circles is None:
                continue

            for circle in np.round(circles[0]).astype(int):
                x, y, r = circle.tolist()
                if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
                    continue
                if y > roi_limit:
                    continue
                if x < roi_left:
                    continue
                fill_ratio = self._circle_fill_ratio(mask, x, y, r)
                if fill_ratio < spec["fill_threshold"]:
                    continue
                detections.append(
                    {
                        "label": label,
                        "x": x,
                        "y": y,
                        "r": r,
                        "score": fill_ratio,
                        "color": spec["draw_color"],
                    }
                )
        return self._deduplicate(detections)

    @staticmethod
    def draw(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        vis = frame.copy()
        for det in detections:
            x, y, r = det["x"], det["y"], det["r"]
            cv2.circle(vis, (x, y), r + 6, det["color"], 2)
            cv2.putText(
                vis,
                f"{det['label']} {det['score']:.2f}",
                (x + 8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                det["color"],
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            vis,
            f"Traffic lights: {len(detections)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
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

            detections = self.detect(frame)
            if frame_index == 0:
                for det in detections:
                    print(
                        f"Frame {frame_index}: {det['label']} at "
                        f"({det['x']}, {det['y']}), r={det['r']}, score={det['score']:.3f}"
                    )

            vis = self.draw(frame, detections)
            if frame_index == 0:
                output_path = PROJECT_ROOT / "Module_5" / "task_5_result.png"
                cv2.imwrite(str(output_path), vis)
                print(f"Saved preview: {output_path}")

            cv2.imshow("Module_5 Task 5", vis)
            frame_index += 1
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = TrafficLightDetector(
        video_path=PROJECT_ROOT / "data" / "city" / "trm.169.007.avi",
    )
    detector.run()
