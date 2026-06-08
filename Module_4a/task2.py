from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SIFTHomographyRansac:
    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.sift = cv2.SIFT_create(nfeatures=800)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def _load_two_frames(self) -> tuple[np.ndarray, np.ndarray]:
        cap = cv2.VideoCapture(str(self.video_path))
        ok1, frame1 = cap.read()
        ok2, frame2 = cap.read()
        cap.release()
        if not ok1 or not ok2:
            raise FileNotFoundError(f"Cannot read two frames from video: {self.video_path}")
        return frame1, frame2

    def _detect_and_match(self, frame1: np.ndarray, frame2: np.ndarray):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.sift.detectAndCompute(gray1, None)
        kp2, des2 = self.sift.detectAndCompute(gray2, None)
        if des1 is None or des2 is None:
            raise RuntimeError("SIFT could not compute descriptors on one of the frames")

        knn_matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < 0.75 * second.distance:
                good_matches.append(first)

        if len(good_matches) < 4:
            raise RuntimeError("Not enough matches for homography estimation")

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        return kp1, kp2, good_matches, src_pts, dst_pts

    @staticmethod
    def _estimate_homography(src_pts: np.ndarray, dst_pts: np.ndarray):
        homography, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            3.0,
            maxIters=2000,
            confidence=0.995,
        )
        if homography is None or mask is None:
            raise RuntimeError("Homography estimation failed")
        return homography, mask.reshape(-1).astype(bool)

    @staticmethod
    def _compute_reprojection_error(src_pts: np.ndarray, dst_pts: np.ndarray, homography: np.ndarray) -> np.ndarray:
        projected = cv2.perspectiveTransform(src_pts, homography)
        return np.linalg.norm(projected.reshape(-1, 2) - dst_pts.reshape(-1, 2), axis=1)

    def analyze(self):
        frame1, frame2 = self._load_two_frames()
        kp1, kp2, matches, src_pts, dst_pts = self._detect_and_match(frame1, frame2)
        homography, inlier_mask = self._estimate_homography(src_pts, dst_pts)
        errors = self._compute_reprojection_error(src_pts, dst_pts, homography)

        result = {
            "frame1": frame1,
            "frame2": frame2,
            "kp1": kp1,
            "kp2": kp2,
            "matches": matches,
            "src_pts": src_pts,
            "dst_pts": dst_pts,
            "homography": homography,
            "inlier_mask": inlier_mask,
            "errors": errors,
            "mean_error": float(errors.mean()),
            "median_error": float(np.median(errors)),
            "mean_inlier_error": float(errors[inlier_mask].mean()),
        }
        return result

    @staticmethod
    def _draw_polygon(frame: np.ndarray, homography: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = frame.shape[:2]
        corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, homography)
        return corners, projected

    def draw_result(self, result: dict) -> np.ndarray:
        frame1 = result["frame1"].copy()
        frame2 = result["frame2"].copy()
        kp1 = result["kp1"]
        kp2 = result["kp2"]
        matches = result["matches"]
        inlier_mask = result["inlier_mask"]
        homography = result["homography"]

        _, projected = self._draw_polygon(frame1, homography)
        cv2.polylines(frame2, [np.int32(projected)], True, (0, 255, 255), 2)

        draw_matches = []
        match_colors = []
        for match, is_inlier in zip(matches, inlier_mask):
            draw_matches.append(match)
            match_colors.append((0, 255, 0) if is_inlier else (0, 0, 255))

        vis = cv2.drawMatches(
            frame1,
            kp1,
            frame2,
            kp2,
            draw_matches[:80],
            None,
            matchColor=(255, 255, 255),
            singlePointColor=(120, 120, 120),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        info_lines = [
            f"Matches after ratio test: {len(matches)}",
            f"RANSAC inliers: {int(inlier_mask.sum())}",
            f"Mean reproj error: {result['mean_error']:.3f}",
            f"Median reproj error: {result['median_error']:.3f}",
            f"Mean inlier error: {result['mean_inlier_error']:.3f}",
        ]
        for idx, line in enumerate(info_lines):
            cv2.putText(
                vis,
                line,
                (20, 30 + idx * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return vis


if __name__ == "__main__":
    analyzer = SIFTHomographyRansac(
        video_path=PROJECT_ROOT / "data" / "city" / "trm.169.007.avi",
    )
    result = analyzer.analyze()
    print(f"Matches after ratio test: {len(result['matches'])}")
    print(f"RANSAC inliers: {int(result['inlier_mask'].sum())}")
    print(f"Mean reprojection error: {result['mean_error']:.6f}")
    print(f"Median reprojection error: {result['median_error']:.6f}")
    print(f"Mean inlier reprojection error: {result['mean_inlier_error']:.6f}")

    vis = analyzer.draw_result(result)
    cv2.imshow("Module_4a Task 2", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
