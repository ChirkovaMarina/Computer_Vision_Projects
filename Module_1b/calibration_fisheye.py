import cv2
import numpy as np
import os
from pathlib import Path

# Load calibration videos from the local calibration directory.
main_path = Path(__file__).resolve().parent / "calib"
images = list()
fs = sorted(os.listdir(main_path))
count = 0
for path in fs:
    cap = cv2.VideoCapture(str(main_path / path))
    while True:
        f, im = cap.read()
        if not f:
            break
        if count % 50 == 0:
            images.append(im)
        count = count + 1
    cap.release()

if not images:
    raise RuntimeError(f"No images were loaded from {main_path}/")


CHECKERBOARD = (7, 7)
CHECKERBOARDSIZE = 100
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)  * CHECKERBOARDSIZE
_img_shape = None
# 3D points in world space.
objpoints = []
# 2D points in the image plane.
imgpoints = []
valid_images = []
for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find chessboard corners in the current frame.
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, append object and image points after corner refinement.
    if ret is True:
        objpoints.append(objp)
        corners = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
        imgpoints.append(corners)
        valid_images.append(img.copy())
N_OK = len(objpoints)
if N_OK == 0:
    raise RuntimeError("Chessboard corners were not detected in any frame")

K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    valid_images[0].shape[:2][::-1],
    K,
    D,
    rvecs,
    tvecs,
    calibration_flags,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
)

plot_image = valid_images[0]
_img_shape = plot_image.shape[:2]
print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")

h, w = plot_image.shape[:2]

new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    K, D, (w, h), np.eye(3), balance=1
)

map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), K, (w, h), cv2.CV_16SC2
)
undistorted_img_1 = cv2.remap(
    plot_image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
)

map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
)
undistorted_img_2 = cv2.remap(
    plot_image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
)

v = np.concatenate((plot_image, undistorted_img_1, undistorted_img_2), axis=1)
cv2.imshow("undistorted", v)
cv2.waitKey(0)
cv2.destroyAllWindows()


colors = [
    (255, 0, 255),
    (255, 0, 0),
    (255, 255, 0),
    (0, 255, 0),
    (0, 255, 255),
    (0, 165, 255),
    (0, 0, 255),
]

object_points_reshaped = objp.astype(np.float64)
for i in range(N_OK):
    plot_image = valid_images[i].copy()
    rvec = rvecs[i]
    tvec = tvecs[i]

    image_points, jacobian = cv2.fisheye.projectPoints(
        object_points_reshaped,
        rvec,
        tvec,
        K,
        D
    )

    projected = image_points.reshape(-1, 2)
    pts = projected.reshape(CHECKERBOARD[1], CHECKERBOARD[0], 2).astype(int)

    for col in range(pts.shape[1]):
        color = colors[col % len(colors)]
        for row in range(pts.shape[0]):
            x, y = pts[row, col]
            cv2.circle(plot_image, (x, y), 9, color, 2)
            cv2.circle(plot_image, (x, y), 2, color, -1)
            if row > 0:
                x_prev, y_prev = pts[row - 1, col]
                cv2.line(plot_image, (x_prev, y_prev), (x, y), color, 2)

    cv2.imshow("projected points", plot_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
