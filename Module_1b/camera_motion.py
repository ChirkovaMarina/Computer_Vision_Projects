import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch
import numpy as np

# Path to the data used for calibration.
DATA_DIR = Path(__file__).resolve().parent / 'calib'

# Checkerboard parameters.
BOARD_DIM = (7, 7)
BOARD_SIZE = 100

# Corner refinement termination criteria.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# World coordinates for checkerboard corners.
obj_ps = np.zeros((1, BOARD_DIM[0] * BOARD_DIM[1], 3), np.float32)
obj_ps[0, :, :2] = np.mgrid[0:BOARD_DIM[0], 0:BOARD_DIM[1]].T.reshape(-1, 2) * BOARD_SIZE

# 3D object points for calibration frames.
obj_points = []
# 2D image points for calibration frames.
img_points = []

images = []  # Collected frames used for calibration.
count = 0  # Running frame counter.

# Sample frames from all calibration videos.
for path in sorted(os.listdir(DATA_DIR)):
    cap = cv2.VideoCapture(str(DATA_DIR / path))
    while True:
        frame_grabbed, img = cap.read()
        if not frame_grabbed:
            break
        if count % 50 == 0:
            images.append(img)
        count += 1
    cap.release()

print('Number of images for calibration: {}'.format(len(images)))
if not images:
    raise RuntimeError(f'No images were loaded from {DATA_DIR}/')


def change_order(points: np.ndarray) -> np.ndarray:
    """Reorder detected chessboard corners when OpenCV returns them flipped."""

    # Coordinates of the 4 outermost corners.
    xs = sorted([points[0][0, 0], points[BOARD_DIM[0] - 1][0, 0], points[-BOARD_DIM[0]][0, 0], points[-1][0, 0]])
    ys = sorted([points[0][0, 1], points[BOARD_DIM[0] - 1][0, 1], points[-BOARD_DIM[0]][0, 1], points[-1][0, 1]])

    if xs.index(points[0][0, 0]) < 2 and ys.index(points[0][0, 1]) < 2:
        # No reordering required.
        return points

    elif xs.index(points[BOARD_DIM[0] - 1][0, 0]) < 2 and ys.index(points[BOARD_DIM[0] - 1][0, 1]) < 2:
        # Points are rotated clockwise.
        return np.rot90(points.reshape(BOARD_DIM[::-1] + (1, 2))).reshape((BOARD_DIM[0] * BOARD_DIM[1], 1, 2))

    elif xs.index(points[-BOARD_DIM[0]][0, 0]) < 2 and ys.index(points[-BOARD_DIM[0]][0, 1]) < 2:
        # Points are rotated counterclockwise.
        return np.rot90(points.reshape(BOARD_DIM[::-1] + (1, 2)), -1).reshape((BOARD_DIM[0] * BOARD_DIM[1], 1, 2))

    # Points are reversed.
    return points[::-1, ...]


for img in images:
    # Convert the frame to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners.
    ret, corners = cv2.findChessboardCorners(gray, BOARD_DIM, cv2.CALIB_CB_ADAPTIVE_THRESH
                                             + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret:
        obj_points.append(obj_ps)

        # Refine pixel coordinates for the detected corners.
        img_ps = cv2.cornerSubPix(gray, np.ascontiguousarray(change_order(corners)), (11, 11), (-1, -1), criteria)
        img_points.append(img_ps)

if not obj_points or not img_points:
    raise RuntimeError('Chessboard corners were not detected in calibration frames')

# Camera calibration.
print('Calibration in progress...')
image_size = (images[0].shape[1], images[0].shape[0])
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)
print('Calibration complete.')

# Number of previous camera positions to plot.
PATH_LEN = 50

# Camera trajectory history.
cam_positions = [[], [], []]

# Create the graph used to render camera motion.
fig: plt.Figure = plt.figure(figsize=(3.4, 3.0))
ax: Axes3D = fig.add_subplot(projection='3d', computed_zorder=False)

# Prepare the data used to draw the checkerboard plane.
x = np.arange(-100, 701, 100)
y = np.arange(-100, 701, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros((9, 9))
colors = np.empty(X.shape, dtype=str)
for y in range(len(Y)):
    for x in range(len(X)):
        colors[y, x] = ('w', 'k')[(x + y) % 2]


class Arrow3D(FancyArrowPatch):
    """Render orientation arrows on the 3D graph."""

    def __init__(self, x0, y0, z0, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = x0, y0, z0
        self._dxdydz = dx, dy, dz

    def do_3d_projection(self):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = x1 + dx, y1 + dy, z1 + dz

        xs, ys, zs = proj3d.proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


for path in sorted(os.listdir(DATA_DIR)):
    cap = cv2.VideoCapture(str(DATA_DIR / path))
    while True:
        frame_grabbed, img = cap.read()
        if not frame_grabbed:
            break

        # Convert the frame to grayscale.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners.
        ret, corners = cv2.findChessboardCorners(gray, BOARD_DIM, cv2.CALIB_CB_ADAPTIVE_THRESH
                                                 + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            # OpenCV sometimes returns corners in the wrong order.
            # Reorder them before pose estimation if necessary.
            corners = np.ascontiguousarray(change_order(corners))

            # Refine pixel coordinates for the detected corners.
            img_ps = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Estimate the checkerboard pose.
            _, rvec, tvec = cv2.solvePnP(obj_ps, img_ps, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)

            # Draw the detected chessboard corners.
            img = cv2.drawChessboardCorners(img, BOARD_DIM, img_ps, ret)

            # Rotation matrix.
            rot_mat = cv2.Rodrigues(rvec)[0]

            # Camera position in checkerboard coordinates.
            cam_pos = -(rot_mat.T @ tvec)

            # Orientation vectors used to draw camera axes.
            ox = np.matmul(rot_mat.T, np.array([300, 0, 0]))
            oy = np.matmul(rot_mat.T, np.array([0, 300, 0]))
            oz = np.matmul(rot_mat.T, np.array([0, 0, 300]))

            # Append the current position to the trajectory.
            for axis in range(3):
                cam_positions[axis].append(cam_pos[axis, 0])

            # Keep the plotted path length bounded.
            if len(cam_positions[0]) > PATH_LEN:
                for axis in range(3):
                    cam_positions[axis] = cam_positions[axis][1:]

            # Current camera coordinates.
            cam_x, cam_y, cam_z = cam_pos[0, 0], cam_pos[1, 0], cam_pos[2, 0]

            # Draw the checkerboard plane.
            ax.plot_surface(X, Y, Z, facecolors=colors, zorder=0)
            # Draw the camera trajectory.
            ax.plot(cam_positions[0], cam_positions[1], cam_positions[2], color='deepskyblue', linewidth=2, zorder=1)
            # Draw the current camera orientation.
            ax.add_artist(  # Z axis
                Arrow3D(cam_x, cam_y, cam_z, oz[0], oz[1], oz[2], arrowstyle='-|>', mutation_scale=8, color='blue'))
            ax.add_artist(  # Y axis
                Arrow3D(cam_x, cam_y, cam_z, oy[0], oy[1], oy[2], arrowstyle='-|>', mutation_scale=8, color='green'))
            ax.add_artist(  # X axis
                Arrow3D(cam_x, cam_y, cam_z, ox[0], ox[1], ox[2], arrowstyle='-|>', mutation_scale=8, color='red'))
            # Draw the current camera position.
            ax.scatter(cam_x, cam_y, cam_z, color='red', linewidths=2, zorder=3)

            # Keep the same scale for all axes.
            ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

            ax.invert_yaxis()  # Invert the Y axis.
            ax.invert_zaxis()  # Invert the Z axis.

            fig.canvas.draw()  # Render the plot.
            w, h = fig.canvas.get_width_height()  # Plot dimensions.
            # Convert the graph image into a NumPy array.
            plot_img = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
            if plot_img.shape[0] > img.shape[0] or plot_img.shape[1] > img.shape[1] // 2:
                max_h = img.shape[0]
                max_w = img.shape[1] // 2
                scale = min(max_h / plot_img.shape[0], max_w / plot_img.shape[1])
                new_size = (
                    max(1, int(plot_img.shape[1] * scale)),
                    max(1, int(plot_img.shape[0] * scale)),
                )
                plot_img = cv2.resize(plot_img, new_size, interpolation=cv2.INTER_AREA)

            # Overlay the plot in the lower-left or lower-right frame corner.
            if img_ps[BOARD_DIM[0] // 2][0, 0] < img.shape[1] // 2:
                img[-plot_img.shape[0]:, -plot_img.shape[1]:, ::-1] = plot_img
            else:
                img[-plot_img.shape[0]:, :plot_img.shape[1], ::-1] = plot_img

            ax.clear()  # Clear the graph for the next frame.

        cv2.imshow('camera motion', img)
        cv2.waitKey(10)

    cap.release()
