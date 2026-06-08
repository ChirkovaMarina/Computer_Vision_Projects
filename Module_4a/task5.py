from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class StationaryObjectMotionDetector:
    def __init__(
        self,
        video_path: Path,
        movement_threshold: float = 2.0,
        stationary_threshold: float = 0.5,
        cell_size: int = 80,
        min_cluster_points: int = 8,
    ):
        self.video_path = video_path
        self.movement_threshold = movement_threshold
        self.stationary_threshold = stationary_threshold
        self.cell_size = cell_size
        self.min_cluster_points = min_cluster_points
        self.detector = cv2.ORB_create(nfeatures=1200)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def _detect_and_match(self, prev_gray: np.ndarray, gray: np.ndarray):
        prev_keypoints, prev_descriptors = self.detector.detectAndCompute(prev_gray, None)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        if prev_descriptors is None or descriptors is None:
            return prev_keypoints, keypoints, []

        knn_matches = self.matcher.knnMatch(prev_descriptors, descriptors, k=2)
        good_matches = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < 0.75 * second.distance:
                good_matches.append(first)
        return prev_keypoints, keypoints, good_matches

    @staticmethod
    def _motion_data(prev_keypoints, keypoints, matches):
        points_prev = []
        points_curr = []
        vectors = []
        for match in matches:
            prev_pt = np.array(prev_keypoints[match.queryIdx].pt, dtype=np.float32)
            curr_pt = np.array(keypoints[match.trainIdx].pt, dtype=np.float32)
            points_prev.append(prev_pt)
            points_curr.append(curr_pt)
            vectors.append(curr_pt - prev_pt)

        if not vectors:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty, empty, empty

        points_prev = np.asarray(points_prev, dtype=np.float32)
        points_curr = np.asarray(points_curr, dtype=np.float32)
        vectors = np.asarray(vectors, dtype=np.float32)
        centers = (points_prev + points_curr) / 2.0
        return points_prev, points_curr, centers, vectors

    def _find_motion_cluster(self, centers: np.ndarray, vectors: np.ndarray):
        if len(vectors) == 0:
            return None

        magnitudes = np.linalg.norm(vectors, axis=1)
        if float(np.median(magnitudes)) > self.stationary_threshold:
            return {"tram_stationary": False}

        moving_mask = magnitudes > self.movement_threshold
        if int(moving_mask.sum()) < self.min_cluster_points:
            return {"tram_stationary": True, "object_detected": False, "magnitudes": magnitudes}

        moving_centers = centers[moving_mask]
        moving_vectors = vectors[moving_mask]
        moving_magnitudes = magnitudes[moving_mask]

        angles = np.arctan2(moving_vectors[:, 1], moving_vectors[:, 0])
        angle_bins = (np.floor(((angles + np.pi) / (2 * np.pi)) * 8).astype(int) % 8)
        direction_counts = np.bincount(angle_bins, minlength=8)
        dominant_bin = int(direction_counts.argmax())
        direction_mask = angle_bins == dominant_bin
        if int(direction_mask.sum()) < self.min_cluster_points:
            return {"tram_stationary": True, "object_detected": False, "magnitudes": magnitudes}

        dir_centers = moving_centers[direction_mask]
        dir_vectors = moving_vectors[direction_mask]
        dir_magnitudes = moving_magnitudes[direction_mask]

        cells = (dir_centers // self.cell_size).astype(int)
        cell_counter = Counter(map(tuple, cells))
        best_cell, best_count = cell_counter.most_common(1)[0]
        if best_count < self.min_cluster_points:
            return {"tram_stationary": True, "object_detected": False, "magnitudes": magnitudes}

        best_mask = np.array([tuple(cell) == best_cell for cell in cells], dtype=bool)
        cluster_centers = dir_centers[best_mask]
        cluster_vectors = dir_vectors[best_mask]
        cluster_magnitudes = dir_magnitudes[best_mask]

        x_min, y_min = np.floor(cluster_centers.min(axis=0) - 15).astype(int)
        x_max, y_max = np.ceil(cluster_centers.max(axis=0) + 15).astype(int)

        return {
            "tram_stationary": True,
            "object_detected": True,
            "bbox": (x_min, y_min, x_max, y_max),
            "cluster_centers": cluster_centers,
            "cluster_vectors": cluster_vectors,
            "cluster_magnitudes": cluster_magnitudes,
            "magnitudes": magnitudes,
        }

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

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_keypoints, keypoints, matches = self._detect_and_match(prev_gray, gray)
            _, _, centers, vectors = self._motion_data(prev_keypoints, keypoints, matches)
            result = self._find_motion_cluster(centers, vectors)

            vis = frame.copy()
            status = "unknown"
            color = (255, 255, 255)
            if result is not None:
                if not result["tram_stationary"]:
                    status = "tram moving"
                    color = (0, 0, 255)
                elif not result.get("object_detected", False):
                    status = "no moving object"
                    color = (0, 255, 0)
                else:
                    status = "moving object detected"
                    color = (0, 0, 255)
                    x_min, y_min, x_max, y_max = result["bbox"]
                    cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), color, 2)
                    for center, vector in zip(result["cluster_centers"], result["cluster_vectors"]):
                        p1 = tuple(np.round(center).astype(int))
                        p2 = tuple(np.round(center + vector).astype(int))
                        cv2.arrowedLine(vis, p1, p2, (0, 255, 255), 1, tipLength=0.3)

            mean_mag = float(np.mean(result["magnitudes"])) if result and "magnitudes" in result else 0.0
            med_mag = float(np.median(result["magnitudes"])) if result and "magnitudes" in result else 0.0
            cluster_points = len(result["cluster_centers"]) if result and result.get("object_detected") else 0
            lines = [
                f"Status: {status}",
                f"Matches: {len(matches)}",
                f"Mean movement: {mean_mag:.3f}px",
                f"Median movement: {med_mag:.3f}px",
                f"Cluster points: {cluster_points}",
            ]
            for idx, line in enumerate(lines):
                cv2.putText(
                    vis,
                    line,
                    (20, 30 + idx * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color if idx == 0 else (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Module_4a Task 5", vis)
            prev_frame = frame
            prev_gray = gray
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = StationaryObjectMotionDetector(
        video_path=PROJECT_ROOT / "data" / "city" / "trm.169.007.avi",
        movement_threshold=2.0,
        stationary_threshold=0.5,
        cell_size=80,
        min_cluster_points=8,
    )
    detector.run()
