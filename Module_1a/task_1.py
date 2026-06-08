from typing import Any
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

BLACK = (0, 0, 0)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
LINE_WIDTH = 5


class WayEstimator:
    """Build the projected vehicle path."""

    def __init__(self, calib_dict: dict[str, Any], ways_length: int):
        self.calib = Calib(calib_dict)
        self.camera = Camera(self.calib)
        self.left_3d_near = Point((-0.76, 5.0, 0))
        self.left_3d_far = Point((-0.76, ways_length, 0))
        self.right_3d_near = Point((0.76, 5.0, 0))
        self.right_3d_far = Point((0.76, ways_length, 0))

    def draw_way(self, img: np.array):
        left_2d_near = self.camera.project_point_3d_to_2d(self.left_3d_near)
        left_2d_far = self.camera.project_point_3d_to_2d(self.left_3d_far)
        right_2d_near = self.camera.project_point_3d_to_2d(self.right_3d_near)
        right_2d_far = self.camera.project_point_3d_to_2d(self.right_3d_far)
        cv2.line(img, pt1=right_2d_near, pt2=right_2d_far, color=BLACK, thickness=LINE_WIDTH)
        cv2.line(img, pt1=left_2d_near, pt2=left_2d_far, color=BLACK, thickness=LINE_WIDTH)
        return img


class Reader:
    def __init__(self, video_path: Path):
        self.video_path = video_path
        par = ["K", "D", "r", "t"]
        calib_reader = CalibReader(
            file_name=str(PROJECT_ROOT / "data" / "city" / "leftImage.yml"),
            param=par,
        )
        calib_dict = calib_reader.read()
        self.way_estimator = WayEstimator(calib_dict, 30)

    def run(self):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            grab_msec = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            cv2.putText(
                frame,
                text=f"GrabMsec: {grab_msec}",
                org=(15, 50),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1.0,
                color=(0, 255, 0),
                thickness=2,
            )
            self.way_estimator.draw_way(frame)
            cv2.imshow("Frame", frame)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = PROJECT_ROOT / "data" / "city" / "trm.169.007.avi"
    reader = Reader(video_path)
    reader.run()
    print("Done!")
