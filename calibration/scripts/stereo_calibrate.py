import cv2
import numpy as np
import glob

# === Parameters ===
video_path = 'stereo_video.avi'
board_size = (8, 6)              # Inner corners per chessboard row/col
square_size = 0.024              # Actual square size in meters

# === Prepare object points ===
objp = np.zeros((board_size[0]*board_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D world points
imgpoints_left = []  # 2D left image points
imgpoints_right = []  # 2D right image points

# === Read video and extract frames ===
cap = cv2.VideoCapture(video_path)
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Split frame into left and right
    h, w = frame.shape[:2]
    left = frame[:, :w//2]
    right = frame[:, w//2:]

    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret_l, corners_l = cv2.findChessboardCorners(gray_left, board_size)
    ret_r, corners_r = cv2.findChessboardCorners(gray_right, board_size)

    if ret_l and ret_r:
        objpoints.append(objp)
        imgpoints_left.append(corners_l)
        imgpoints_right.append(corners_r)

        # Optionally draw and save
        cv2.drawChessboardCorners(left, board_size, corners_l, ret_l)
        cv2.drawChessboardCorners(right, board_size, corners_r, ret_r)
        vis = np.hstack((left, right))
        cv2.imshow("Chessboard Detection", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

# === Run stereo calibration ===
ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    None, None, None, None,
    gray_left.shape[::-1],
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-5),
    flags=cv2.CALIB_FIX_INTRINSIC
)

print("Calibration RMS error:", ret)

