import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Checkerboard dimensions.
CHECKERBOARD = (7, 7)
CHECKERBOARDSIZE = 100
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# 3D object points for each checkerboard frame.
objpoints = []
# 2D image points for each checkerboard frame.
imgpoints = []
# World coordinates for checkerboard corners.
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * CHECKERBOARDSIZE 

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

for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find chessboard corners in the current frame.
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret is True:
        objpoints.append(objp)
        # Refine pixel coordinates for the detected corners.
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # Draw and display the detected corners.
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    cv2.imshow('img', img)
    cv2.waitKey(0)
cv2.destroyAllWindows()

image = images[0]
gray0 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
h, w = gray0.shape[:2]
"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
if not objpoints or not imgpoints:
    raise RuntimeError("Chessboard corners were not detected in any frame")

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray0.shape[::-1], None, None)
print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)


image = images[0]
map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, np.eye(3), mtx, (w, h), cv2.CV_16SC2)
undistorted_img = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

newCameraMatrix, validPixROI = cv2.getOptimalNewCameraMatrix(mtx, dist,(w, h), 1, (w, h))
undistorted_image = cv2.undistort(
    image, mtx, dist, None, newCameraMatrix
)
roi_x, roi_y, roi_w, roi_h = validPixROI
cv2.rectangle(undistorted_image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 1)

v = np.concatenate((image, undistorted_img, undistorted_image), axis=1)
cv2.imshow("undistorted_remap", v)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Generate a regular 3D grid for reprojection visualization.
grid_size, square_size = [10, 10], 0.5
object_points = np.zeros([grid_size[0] * grid_size[1], 3])
mx, my = [(grid_size[0] - 1) * square_size / 2, (grid_size[1] - 1) * square_size / 2]
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        object_points[i * grid_size[0] + j] = [i * square_size - mx, j * square_size - my, 0]
# Use the estimated camera parameters.
intrinsic = mtx
rvec = rvecs[0]
tvec = tvecs[0]
# Project the points with the calibrated model.
image_points, jacobian = cv2.projectPoints(object_points, rvec, tvec, intrinsic, dist)
# Plot the projected points with Matplotlib.
plt.scatter(*zip(*image_points[:, 0, :]), marker='.')
plt.axis('equal')
plt.grid()
plt.show()
