import cv2
import numpy as np
import glob
import pickle

# -------------------- Calibration Settings -------------------- #
# Number of inner corners per chessboard row and column.
# For a chessboard with 10x7 squares, there are 9x6 inner corners.
pattern_size = (10, 7)  # (columns, rows)

# Real-world size of each square on your chessboard (in meters)
square_size = 0.020  # e.g., 20 mm

# Prepare a single set of object points in the real-world coordinate system
# (0,0,0), (1,0,0), (2,0,0), ... scaled by the square size.
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store the object points and image points from all images.
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Termination criteria for sub-pixel corner refinement.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# -------------------- Load Calibration Images -------------------- #
# Make sure your calibration images are stored in the folder "calibration_images"
images = glob.glob('CV Viaduct/calibration_images/*.jpg')
if not images:
    print("No calibration images found in 'calibration_images' folder. Exiting.")
    exit()

print(f"Found {len(images)} calibration images.")

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Could not load image: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ret:
        # Save the real world coordinates and image points.
        objpoints.append(objp)

        # Refine corner locations to sub-pixel accuracy.
        corners_subpix = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners_subpix)

        # Draw and display the corners for visual feedback.
        cv2.drawChessboardCorners(img, pattern_size, corners_subpix, ret)
        # Half the original image size
        img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
        cv2.imshow('Detected Corners', img)
        cv2.waitKey(500)

    else:
        print(f"Chessboard not detected in {fname}")

cv2.destroyAllWindows()

# -------------------- Camera Calibration -------------------- #
if len(objpoints) < 10:
    print("Not enough valid calibration images detected. Need at least 10 images.")
    exit()

# Calibrate the camera using the detected points.
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("Calibration successful:", ret)
print("\nCamera Matrix:")
print(camera_matrix)
print("\nDistortion Coefficients:")
print(dist_coeffs)

# -------------------- Reprojection Error Calculation -------------------- #
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
mean_error /= len(objpoints)
print(f"\nMean Reprojection Error: {mean_error:.4f}")

# -------------------- Save Calibration Data -------------------- #
calibration_data = {
    'camera_matrix': camera_matrix,
    'dist_coeffs': dist_coeffs,
    'reprojection_error': mean_error
}

with open('calibration_data.pkl', 'wb') as f:
    pickle.dump(calibration_data, f)

print("\nCalibration data saved to 'calibration_data.pkl'.")
