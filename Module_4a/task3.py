from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class KeypointMotionClassifier:
    def __init__(self, video_path: Path, movement_threshold: float = 1.5):
        self.video_path = video_path
        self.movement_threshold = movement_threshold
        self.detector = cv2.ORB_create(nfeatures=800)
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
    def _motion_vectors(prev_keypoints, keypoints, matches) -> np.ndarray:
        vectors = []
        for match in matches:
            prev_pt = np.array(prev_keypoints[match.queryIdx].pt, dtype=np.float32)
            curr_pt = np.array(keypoints[match.trainIdx].pt, dtype=np.float32)
            vectors.append(curr_pt - prev_pt)
        if not vectors:
            return np.empty((0, 2), dtype=np.float32)
        return np.asarray(vectors, dtype=np.float32)

    def _classify_motion(self, vectors: np.ndarray) -> tuple[str, float, float, np.ndarray]:
        if len(vectors) == 0:
            return "unknown", 0.0, 0.0, np.zeros(2, dtype=np.float32)

        magnitudes = np.linalg.norm(vectors, axis=1)
        median_magnitude = float(np.median(magnitudes))
        mean_magnitude = float(np.mean(magnitudes))
        dominant_vector = np.median(vectors, axis=0)
        status = "moving" if median_magnitude > self.movement_threshold else "stopped"
        return status, mean_magnitude, median_magnitude, dominant_vector

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
            vectors = self._motion_vectors(prev_keypoints, keypoints, matches)
            status, mean_magnitude, median_magnitude, dominant_vector = self._classify_motion(vectors)

            vis = cv2.drawMatches(
                prev_frame,
                prev_keypoints,
                frame,
                keypoints,
                matches[:80],
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )

            lines = [
                f"Status: {status}",
                f"Matches: {len(matches)}",
                f"Mean movement: {mean_magnitude:.3f}px",
                f"Median movement: {median_magnitude:.3f}px",
                f"Dominant vector: ({dominant_vector[0]:.3f}, {dominant_vector[1]:.3f})",
            ]
            color = (0, 255, 0) if status == "stopped" else (0, 0, 255)
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

            cv2.imshow("Module_4a Task 3", vis)
            prev_frame = frame
            prev_gray = gray

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    classifier = KeypointMotionClassifier(
        video_path=PROJECT_ROOT / "data" / "city" / "trm.169.007.avi",
        movement_threshold=1.5,
    )
    classifier.run()
