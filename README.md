# CV Viaduct Project

A research project conducted at Nanyang Technological University (NTU), Singapore.

This project involves detecting markers and calibrating an iPhone camera (iPhone 13 Pro Max Main Wide Camera) using OpenCV and ArUco markers. It includes scripts for processing images and calculating camera parameters for structural monitoring applications.

## Project Context

This research is conducted as part of computer vision studies at Nanyang Technological University. The project aims to develop reliable CV techniques for segment displacement of viaducts. This project is built upon a real on-going viaduct construction project within NTU campus, Singapore.

## Project Structure

- `MarkerDetect.py`: Module for detecting and tracking ArUco markers, calculating relative poses between markers and camera.
- `iphoneCameraCali.py`: Script for calibrating the iPhone 13 Pro Max camera using a chessboard pattern, generating intrinsic and extrinsic parameters.

## Requirements

- Python 3.8 or higher
- OpenCV 4.5+
- NumPy 1.20+
- Pillow 8.0+
- Matplotlib (for visualization)
- SciPy (for mathematical operations)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cv-viaduct.git
   cd cv-viaduct
   ```
   
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Camera Calibration:
   ```bash
   python iphoneCameraCali.py --input_dir ./calibration_images --output_dir ./camera_parameters --grid_size 9x6 --square_size 25
   ```
   Parameters:
   - `--input_dir`: Directory containing calibration images
   - `--output_dir`: Directory to save calibration parameters
   - `--grid_size`: Dimensions of the checkerboard (inner corners)
   - `--square_size`: Size of checkerboard squares in millimeters

2. Marker Detection:
   ```bash
   python MarkerDetect.py --image ./test_images/sample.jpg --parameters ./camera_parameters/camera_params.json --marker_size 100
   ```
   Parameters:
   - `--image`: Path to the image for marker detection
   - `--parameters`: Path to camera calibration parameters
   - `--marker_size`: Size of ArUco markers in millimeters (default: 100)

3. Batch Processing (for multiple images):
   ```bash
   python MarkerDetect.py --input_dir ./test_images --parameters ./camera_parameters/camera_params.json --output_dir ./results
   ```

## Acknowledgments

This project is conducted at Nanyang Technological University, Singapore. We acknowledge the support and resources provided by NTU for this research.

## License

This project is licensed under the MIT License.