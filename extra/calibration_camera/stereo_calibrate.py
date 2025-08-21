import cv2
import numpy as np
import glob
import yaml
import sys

# --- Checkerboard settings ---
checkerboard_size = (7, 9)  # inner corners (columns, rows)
square_size = 0.015          # meters per square

# Prepare object points
objp = np.zeros((checkerboard_size[1]*checkerboard_size[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store points
objpoints = []      # 3D points
imgpoints_left = [] # 2D points in left images
imgpoints_right = []# 2D points in right images

# --- Load images ---
left_images = sorted(glob.glob("left_*.png"))
right_images = sorted(glob.glob("right_*.png"))

if len(left_images) == 0 or len(right_images) == 0:
    print("No images found! Make sure left_*.png and right_*.png exist in the folder.")
    sys.exit(1)

for l_img, r_img in zip(left_images, right_images):
    img_l = cv2.imread(l_img, cv2.IMREAD_GRAYSCALE)
    img_r = cv2.imread(r_img, cv2.IMREAD_GRAYSCALE)

    ret_l, corners_l = cv2.findChessboardCorners(img_l, checkerboard_size, None)
    ret_r, corners_r = cv2.findChessboardCorners(img_r, checkerboard_size, None)

    if ret_l and ret_r:
        objpoints.append(objp)

        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2_l = cv2.cornerSubPix(img_l, corners_l, (11,11), (-1,-1), term)
        corners2_r = cv2.cornerSubPix(img_r, corners_r, (11,11), (-1,-1), term)

        imgpoints_left.append(corners2_l)
        imgpoints_right.append(corners2_r)

        # --- Show detected corners ---
        img_l_color = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR)
        img_r_color = cv2.cvtColor(img_r, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(img_l_color, checkerboard_size, corners2_l, ret_l)
        cv2.drawChessboardCorners(img_r_color, checkerboard_size, corners2_r, ret_r)

        combined = np.hstack((img_l_color, img_r_color))
        cv2.imshow("Corners Detected (Left | Right)", combined)
        cv2.waitKey(500)  # 0.5 second
    else:
        print(f"Chessboard not found in {l_img} or {r_img}")

cv2.destroyAllWindows()

if len(imgpoints_left) == 0:
    print("No valid chessboard corners detected. Exiting.")
    sys.exit(1)

img_shape = img_l.shape[::-1]  # width, height from last valid image

# --- Calibrate each camera individually ---
ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
    objpoints, imgpoints_left, img_shape, None, None)
ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
    objpoints, imgpoints_right, img_shape, None, None)

# --- Stereo calibration ---
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

ret_s, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtx_l, dist_l, mtx_r, dist_r,
    img_shape, criteria=(cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 100, 1e-5),
    flags=flags)

# --- Stereo rectification ---
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtx_l, dist_l, mtx_r, dist_r, img_shape, R, T, alpha=0)

# --- Generate rectification maps ---
map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, img_shape, cv2.CV_16SC2)
map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, img_shape, cv2.CV_16SC2)

# --- Save maps to YAML ---
data = {
    'map1_l': map1_l.tolist(),
    'map2_l': map2_l.tolist(),
    'map1_r': map1_r.tolist(),
    'map2_r': map2_r.tolist(),
    'P1': P1.tolist(),
    'P2': P2.tolist(),
    'R': R.tolist(),
    'Q': Q.tolist()
}

with open('stereo_rectify_maps.yml', 'w') as f:
    yaml.dump(data, f)

print("Saved stereo_rectify_maps.yml successfully!")
