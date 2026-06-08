import cv2
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRCCAM_ROOT = PROJECT_ROOT / "srccam"
if str(SRCCAM_ROOT) not in sys.path:
    sys.path.insert(0, str(SRCCAM_ROOT))

from srccam.load_calib import CalibReader
from srccam.calib import Calib
from srccam.camera import Camera
from srccam.point import Point3d as Point


class PointsCounter:

    def __init__(self, calib_dict, point_importance):
        self.prev_points = np.array([])
        self.calib = Calib(calib_dict)
        self.camera = Camera(self.calib)
        self.gps = []
        self.speed = 0 # Speed in km/h.
        self.time_per_shot = 0.02 # Time interval between shots.
        self.left_2d_far = self.camera.project_point_3d_to_2d(Point((-1.5, 12, 0)))
        self.right_2d_far = self.camera.project_point_3d_to_2d(Point((1.5, 12, 0)))
        self.A_inv = self.count_matrix_for_2d_3d_projection()
        self.points_importance = point_importance  # Point threshold level.
        self.yaw = 0
        self.image_shape = None

    def count_point_moving(self, Ri, t_ii1, yaw_i: float = 0.0):
        """
        Process points on the ground plane.

        Arguments:
        Ri -- points on the source frame,
        Ti -- starting path point,
        tii1 -- displacement from GPS,
        yawi -- yaw angle from GPS (default 0)
        """

        Rz = np.array([  # Rotation matrix built from GPS yaw.
            [np.cos(yaw_i), -np.sin(yaw_i), 0],
            [np.sin(yaw_i), np.cos(yaw_i), 0],
            [0, 0, 1],
        ])

        RR_i = []
        t_ii1 = np.array([[t_ii1[0]], [t_ii1[1]], [t_ii1[2]]])  # Translation vector.
        for i in Ri:
            i.vec = Rz @ (i.vec + t_ii1)  # Shift current points.
            a = self.camera.project_point_3d_to_2d(i)

            # Make sure the projection stays inside the image bounds.
            image_height, image_width = self.image_shape[:2]
            if 0 <= a[0] < image_width and 0 <= a[1] < image_height:
                RR_i.append(a)
        return RR_i

    def speed_metre_per_sec(self):
        return self.speed * 10 / 36

    def perv_points_projection_to_new(self, img):
        """Draw old and new points on the image."""
        self.image_shape = img.shape
        # Apply Harris to the current image and clip extra points.
        new_Harris = self.apply_Harris(img)
        new_Harris[:self.left_2d_far[1], :] = 0
        new_Harris[:, :self.left_2d_far[0]] = 0
        new_Harris[:, self.right_2d_far[0]:] = 0

        # Work with points from the previous frame.
        if self.prev_points.size > 0:
            a = np.array(
                self.count_point_moving(
                    self.get_3d_points_on_land(self.prev_points),
                    [0, self.time_per_shot * self.speed_metre_per_sec(), 0.],
                    self.yaw))
            if a.size != 0:
                # Draw predicted points in blue.
                img[a[:, 1], a[:, 0]] = [255, 0, 0]
            # Draw previous points in green.
            img[self.prev_points
                > self.points_importance * self.prev_points.max()] = [0, 255, 0]
        # Draw current points in red.
        img[new_Harris > self.points_importance * new_Harris.max()] = [0, 0, 255]

        # Save current points for the next frame.
        self.prev_points = new_Harris
        return img

    def get_3d_points_on_land(self, new_Harris):
        """Filter out points that do not belong to the selected ground region."""
        # Get coordinates of points that pass the threshold.
        points = np.argwhere(new_Harris
            > self.points_importance * new_Harris.max())
        # Convert them to 3D coordinates.
        points = np.apply_along_axis(
            self.reproject_point_2d_to_3d_on_floor, 1, points)
        return points

    @staticmethod
    def get_A_from_P_on_floor(P: np.ndarray) -> np.ndarray:
        """
        The first two columns stay unchanged,
        and the last two columns are combined.
        """
        h = 0  # Projection is on the ground plane, so height is zero.
        A = np.zeros((3, 3))
        A[0, 0], A[0, 1], A[0, 2] = P[0, 0], P[0, 1], h * P[0, 2] + P[0, 3]
        A[1, 0], A[1, 1], A[1, 2] = P[1, 0], P[1, 1], h * P[1, 2] + P[1, 3]
        A[2, 0], A[2, 1], A[2, 2] = P[2, 0], P[2, 1], h * P[2, 2] + P[2, 3]
        return A

    def reproject_point_2d_to_3d_on_floor(self, point2d: None):
        """
        Reproject 2D points into 3D coordinates.

        Only for points that lie on the ground plane.
        """
        if point2d is None:
            point2d = []
        h = 0  # Projection is on the ground plane, so height is zero.
        p_ = np.asarray(self.A_inv @ Point((point2d[0], point2d[1], 1)).vec, dtype=float).reshape(-1)
        error = np.asarray(abs(self.calib.t) / 2, dtype=float).reshape(-1)  # Conversion uncertainty between coordinate systems.
        x = float(p_[0] / p_[2] + error[0])
        y = float(p_[1] / p_[2] + error[1])
        z = h
        return Point((x, y, z))

    def count_matrix_for_2d_3d_projection(self):
        """Precompute matrices required for 2D-to-3D projection."""
        R = self.calib.cam_to_vr @ self.calib.r  # Change axis order.
        affine_matrix = np.concatenate((R, -R @ self.calib.t), 1)
        P = self.calib.K @ affine_matrix
        A = self.get_A_from_P_on_floor(P)
        A_inv = np.linalg.inv(A)
        return A_inv

    def apply_Harris(self, img):
        """
        Harris detector.

        Accepts a BGR image and returns a grayscale corner response map.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.15)
        dst = cv2.dilate(dst, None)
        return dst


class Reader:
    """Video stream processing."""

    def __init__(self, video_path: Path, speed_kmh: float = 10.0, yaw_rad: float = 0.0):
        self.video_path = video_path
        par = ['K', 'D', 'r', 't']
        calib_reader = CalibReader(
            file_name=str(PROJECT_ROOT / 'data' / 'city' / 'leftImage.yml'),
            param=par,
        )
        calib_dict = calib_reader.read()
        self.counter = PointsCounter(calib_dict, 0.20)
        self.counter.speed = speed_kmh
        self.counter.yaw = yaw_rad

    def run(self):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            grab_msec = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            cv2.putText(frame, f'GrabMsec: {grab_msec}', (15, 35),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f'Speed km/h: {self.counter.speed:.1f}', (15, 60),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f'Yaw rad: {self.counter.yaw:.3f}', (15, 85),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
            self.counter.perv_points_projection_to_new(frame)
            cv2.imshow("Task 1", frame)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    s = Reader(
        video_path=PROJECT_ROOT / 'data' / 'city' / 'trm.169.007.avi',
        speed_kmh=10.0,
        yaw_rad=0.0,
    )
    s.run()
    print('Done!')
