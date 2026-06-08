from pathlib import Path
import math

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REJECT_DEGREE_TH = 4.0


class HorizonByVanishingPoint:
    def __init__(self, video_path: Path):
        self.video_path = video_path

    @staticmethod
    def _filter_lines(lines, frame_height: int):
        filtered = []
        if lines is None:
            return filtered

        for line in lines:
            x1, y1, x2, y2 = map(int, line[0])
            dx = x2 - x1
            dy = y2 - y1
            if x1 != x2:
                m = dy / dx
            else:
                m = 1e8
            c = y2 - m * x2
            theta = math.degrees(math.atan(m)) if abs(m) < 1e7 else 90.0
            length = math.hypot(y2 - y1, x2 - x1)
            if length < 100:
                continue
            if max(y1, y2) < frame_height * 0.35:
                continue
            # Rails in perspective are long, steep, but not close to vertical.
            if not (20.0 <= abs(theta) <= 75.0):
                continue
            filtered.append([x1, y1, x2, y2, m, c, length, theta])

        if len(filtered) > 20:
            filtered = sorted(filtered, key=lambda item: item[-1], reverse=True)[:20]
        return filtered

    def _get_lines(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 1)
        edges = cv2.Canny(blur, 40, 255)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=80, maxLineGap=20)
        rail_candidates = self._filter_lines(lines, frame.shape[0])
        left = [line for line in rail_candidates if line[4] < -0.15]
        right = [line for line in rail_candidates if line[4] > 0.15]
        return left, right

    @staticmethod
    def _line_intersection(line1, line2):
        x1, y1, x2, y2 = line1[:4]
        x3, y3, x4, y4 = line2[:4]
        a1 = y2 - y1
        b1 = x1 - x2
        c1 = a1 * x1 + b1 * y1
        a2 = y4 - y3
        b2 = x3 - x4
        c2 = a2 * x3 + b2 * y3
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-8:
            return None
        x = (b2 * c1 - b1 * c2) / det
        y = (a1 * c2 - a2 * c1) / det
        return [x, y]

    def _get_vanishing_point(self, left_lines, right_lines, frame_shape):
        intersections = []
        frame_width, frame_height = frame_shape[1], frame_shape[0]
        for left in left_lines:
            for right in right_lines:
                point = self._line_intersection(left, right)
                if point is None:
                    continue
                x, y = point
                if not (-frame_width <= x <= 2 * frame_width):
                    continue
                if not (-frame_height <= y <= 2 * frame_height):
                    continue
                intersections.append(point)

        if not intersections:
            return None, []

        intersections = np.asarray(intersections, dtype=float)
        # "Most saturated" horizon level: densest horizontal band of intersections.
        y_values = intersections[:, 1]
        bins = np.arange(-frame_height, 2 * frame_height + 5, 5)
        hist, edges = np.histogram(y_values, bins=bins)
        best_bin = int(np.argmax(hist))
        y_low, y_high = edges[best_bin], edges[best_bin + 1]
        mask = (y_values >= y_low) & (y_values < y_high)
        cluster = intersections[mask]
        vp_x = float(np.median(cluster[:, 0]))
        vp_y = float(np.median(cluster[:, 1]))
        return [vp_x, vp_y], intersections.tolist()

    def draw_result(self, frame: np.ndarray, left_lines, right_lines, intersections, vanishing_point):
        vis = frame.copy()
        for line in left_lines + right_lines:
            cv2.line(vis, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 2)

        for x, y in intersections[:100]:
            if -2000 < x < 2000 and -2000 < y < 2000:
                cv2.circle(vis, (int(x), int(y)), 2, (0, 200, 200), -1)

        if vanishing_point is not None:
            vp_x, vp_y = int(vanishing_point[0]), int(vanishing_point[1])
            cv2.circle(vis, (vp_x, vp_y), 5, (0, 0, 255), -1)
            cv2.line(vis, (0, vp_y), (frame.shape[1] - 1, vp_y), (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"VP: ({vp_x}, {vp_y})",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                f"Horizon y = {vp_y}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                vis,
                "Vanishing point not found",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return vis

    def run(self):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            left_lines, right_lines = self._get_lines(frame)
            vanishing_point, intersections = self._get_vanishing_point(left_lines, right_lines, frame.shape)
            vis = self.draw_result(frame, left_lines, right_lines, intersections, vanishing_point)
            cv2.imshow("Module_5 Task 3", vis)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = HorizonByVanishingPoint(
        video_path=PROJECT_ROOT / "data" / "city" / "trm.169.007.avi",
    )
    detector.run()
