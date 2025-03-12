import cv2
import numpy as np
from PIL import Image
import glob
import os
import time
from pathlib import Path

# Import ArUco module explicitly
from cv2 import aruco

# Constants and configuration
# ArUco configuration
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

# Calibrated iphone 13pm camera parameters
CAMERA_MATRIX = np.array([
    [3.03671333e+03, 0.00000000e+00, 1.47645324e+03],
    [0.00000000e+00, 3.04258016e+03, 1.99270842e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
], dtype=np.float64)

DIST_COEFFS = np.array([
    2.52297387e-01, -1.67839933e+00, -4.19392270e-03, -9.71453842e-03, 7.25147589e+00
], dtype=np.float64)
MARKER_LENGTH = 0.28  # meters

# Detector parameters configuration
parameters.adaptiveThreshWinSizeMin = 5
parameters.adaptiveThreshWinSizeMax = 35
parameters.minMarkerPerimeterRate = 0.03
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

# Default paths
DEFAULT_IMAGE_FOLDER = os.path.join("raw images")
DEFAULT_RESULTS_FOLDER = os.path.join("results")

# Required marker IDs
REQUIRED_MARKERS = [0, 4]


def load_and_preprocess(image_path, enhance_contrast=True):
    """
    Load image and preprocess with optional contrast enhancement

    Args:
        image_path (str): Path to the input image
        enhance_contrast (bool): Whether to enhance contrast

    Returns:
        tuple: (preprocessed image, new camera matrix) or (None, None) if loading fails
    """
    try:
        # Directly load image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return None, None

        # Enhance contrast
        if enhance_contrast:
            # Convert to LAB color space (L=lightness, A=green-red, B=blue-yellow)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            # Split channels
            l, a, b = cv2.split(lab)
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to lightness channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            # Merge channels
            enhanced_lab = cv2.merge((cl, a, b))
            # Convert back to BGR
            img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Undistort
        h, w = img.shape[:2]
        new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(
            CAMERA_MATRIX, DIST_COEFFS, (w, h), 1, (w, h))
        undistorted_img = cv2.undistort(
            img, CAMERA_MATRIX, DIST_COEFFS, None, new_cam_matrix)
        return undistorted_img, new_cam_matrix

    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None, None


def rotation_matrix_to_euler(R):
    """
    Convert rotation matrix to Euler angles (degrees) with singularity handling

    Args:
        R (numpy.ndarray): 3x3 rotation matrix

    Returns:
        numpy.ndarray: Euler angles in degrees [roll, pitch, yaw]
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    if sy > 1e-6:  # Non-singular case
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:  # Singular case
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.degrees([x, y, z])


def create_transform_matrix(rvec, tvec):
    """
    Create a 4x4 transformation matrix from rotation and translation vectors

    Args:
        rvec (numpy.ndarray): Rotation vector
        tvec (numpy.ndarray): Translation vector

    Returns:
        numpy.ndarray: 4x4 transformation matrix
    """
    R, _ = cv2.Rodrigues(rvec)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = tvec.flatten()
    return M


def calculate_relative_pose(ref_rvec, ref_tvec, tar_rvec, tar_tvec):
    """
    Calculate relative pose between two markers

    Args:
        ref_rvec (numpy.ndarray): Reference marker rotation vector
        ref_tvec (numpy.ndarray): Reference marker translation vector
        tar_rvec (numpy.ndarray): Target marker rotation vector
        tar_tvec (numpy.ndarray): Target marker translation vector

    Returns:
        numpy.ndarray: 4x4 transformation matrix representing relative pose
    """
    ref_in_cam = create_transform_matrix(ref_rvec, ref_tvec)
    tar_in_cam = create_transform_matrix(tar_rvec, tar_tvec)
    return np.linalg.inv(ref_in_cam) @ tar_in_cam


def analyze_image(image_path, visualize=False, enhance_contrast=True, output_folder=None):
    """
    Detect ArUco markers and analyze their relative positions

    Args:
        image_path (str): Path to the input image
        visualize (bool): Whether to visualize and save results
        enhance_contrast (bool): Whether to enhance contrast
        output_folder (str): Path to save visualization results

    Returns:
        dict: Analysis results or None if processing fails
    """
    start_time = time.time()
    print(f"Processing: {image_path}")

    # Load and preprocess image
    img, new_cam_matrix = load_and_preprocess(
        image_path, enhance_contrast=enhance_contrast)

    if img is None:
        print(f"Failed to process {image_path}")
        return None

    # Detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        corners, ids, rejected = aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters)
    except Exception as e:
        print(f"Error during marker detection: {str(e)}")
        return None

    if ids is None or len(ids) == 0:
        print(f"No markers detected in {image_path}")
        return None

    # Check for required markers
    flat_ids = ids.flatten()
    missing_markers = [
        marker for marker in REQUIRED_MARKERS if marker not in flat_ids]
    if missing_markers:
        print(f"Required markers {missing_markers} not found in {image_path}")
        print(f"Detected marker IDs: {flat_ids}")
        return None

    try:
        # Pose estimation
        ref_idx = np.where(ids == REQUIRED_MARKERS[0])[0][0]
        tar_idx = np.where(ids == REQUIRED_MARKERS[1])[0][0]

        # Use new_cam_matrix and zero distortion coefficients for pose estimation
        # since we're working with an already undistorted image
        zero_dist_coeffs = np.zeros(5)

        rvec_ref, tvec_ref, _ = aruco.estimatePoseSingleMarkers(
            corners[ref_idx], MARKER_LENGTH, new_cam_matrix, zero_dist_coeffs)
        rvec_tar, tvec_tar, _ = aruco.estimatePoseSingleMarkers(
            corners[tar_idx], MARKER_LENGTH, new_cam_matrix, zero_dist_coeffs)

        # Calculate relative transformation
        rel_transform = calculate_relative_pose(rvec_ref[0], tvec_ref[0],
                                                rvec_tar[0], tvec_tar[0])

        # Extract metrics
        rel_position = rel_transform[:3, 3]
        x_displacement = rel_position[0]
        y_displacement = rel_position[1]
        distance = np.linalg.norm(rel_position)
        euler_angles = rotation_matrix_to_euler(rel_transform[:3, :3])

        # Visualization
        if visualize:
            vis_img = img.copy()
            aruco.drawDetectedMarkers(vis_img, corners, ids)
            for i, (rvec, tvec) in enumerate([(rvec_ref[0], tvec_ref[0]), (rvec_tar[0], tvec_tar[0])]):
                cv2.drawFrameAxes(vis_img, new_cam_matrix, zero_dist_coeffs,
                                  rvec, tvec, MARKER_LENGTH/2)

            # Create output folder if it doesn't exist
            if output_folder is None:
                output_folder = os.path.join(
                    os.path.dirname(image_path), "results")
            os.makedirs(output_folder, exist_ok=True)

            # Save visualization
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_folder, f"result_{filename}")
            cv2.imwrite(output_path, vis_img)

            # Display
            cv2.imshow('Analysis', cv2.resize(vis_img, (0, 0), fx=0.5, fy=0.5))
            cv2.waitKey(1000)

        # Results
        processing_time = time.time() - start_time
        print(
            f"\nResults for {image_path} (processed in {processing_time:.2f}s):")
        print(f"  X Displacement: {x_displacement:.3f}m")
        print(f"  Y Displacement: {y_displacement:.3f}m")

        return {
            "image_path": image_path,
            "x_displacement": x_displacement,
            "y_displacement": y_displacement,
            "processing_time": processing_time
        }
    except Exception as e:
        print(f"Error during pose estimation: {str(e)}")
        return None


def process_images(image_folder=DEFAULT_IMAGE_FOLDER, results_folder=DEFAULT_RESULTS_FOLDER,
                   visualize=True, enhance_contrast=True, file_ext=".jpg"):
    """
    Process all images in the specified folder

    Args:
        image_folder (str): Folder containing images to process
        results_folder (str): Folder to save visualization results
        visualize (bool): Whether to visualize and save results
        enhance_contrast (bool): Whether to enhance contrast
        file_ext (str): File extension to look for

    Returns:
        list: List of analysis results with X and Y displacements
    """
    results = []

    # Use pathlib for better path handling
    image_folder_path = Path(image_folder)
    results_folder_path = Path(results_folder)

    # Ensure image folder exists
    if not image_folder_path.exists():
        print(f"Image folder not found: {image_folder}")
        print("Creating folder, please place images there.")
        image_folder_path.mkdir(parents=True, exist_ok=True)
        return results

    # Process each image
    image_paths = list(image_folder_path.glob(f"*{file_ext}"))

    if not image_paths:
        print(f"No {file_ext} images found in {image_folder}")
        return results

    print(f"Found {len(image_paths)} images to process")

    # Ensure results folder exists
    results_folder_path.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        result = analyze_image(
            str(img_path),
            visualize=visualize,
            enhance_contrast=enhance_contrast,
            output_folder=str(results_folder_path)
        )
        if result:
            results.append(result)

    return results


def main():
    """Main function to process images and display X-Y displacements"""
    start_time = time.time()

    # Process the images
    results = process_images(
        image_folder=DEFAULT_IMAGE_FOLDER,
        results_folder=DEFAULT_RESULTS_FOLDER,
        visualize=True,
        enhance_contrast=True
    )

    # Close all windows when done
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Report results
    total_time = time.time() - start_time
    print(
        f"\nProcessed {len(results)} images successfully in {total_time:.2f}s")

    if results:
        # Display X-Y displacement summary
        print("\nX-Y Displacement Summary:")
        print("-------------------------")
        print("Image                   |  X (m)   |  Y (m)   ")
        print("-------------------------+----------+----------")

        for result in results:
            img_name = os.path.basename(result["image_path"])
            x_disp = result["x_displacement"]
            y_disp = result["y_displacement"]
            print(f"{img_name:25} | {x_disp:8.3f} | {y_disp:8.3f}")

        # Calculate averages
        avg_x = sum(r["x_displacement"] for r in results) / len(results)
        avg_y = sum(r["y_displacement"] for r in results) / len(results)
        print("-------------------------+----------+----------")
        print(f"Average                 | {avg_x:8.3f} | {avg_y:8.3f}")


if __name__ == "__main__":
    main()
