# CV Viaduct Project

A research project conducted at Nanyang Technological University (NTU), Singapore.

This project involves detecting markers and calibrating an iPhone camera (iPhone 13 Pro Max Main Wide Camera) using OpenCV and ArUco markers. It includes scripts for processing images and calculating camera parameters for structural monitoring applications.

## Project Context

This research is conducted as part of computer vision studies at Nanyang Technological University. The project aims to develop reliable camera calibration techniques for structural monitoring of viaducts and similar infrastructure.

## Project Structure

- `MarkerDetect.py`: Contains functions for detecting ArUco markers in images and calculating relative poses.
- `iphoneCameraCali.py`: Contains functions for calibrating the iPhone camera using a chessboard pattern.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Pillow

## Installation

1. Clone the repository.
2. Install the required packages using `pip install -r requirements.txt`.

## Usage

1. Run `iphoneCameraCali.py` to calibrate the camera:
   ```
   python iphoneCameraCali.py --input_dir path/to/calibration/images --output_dir path/to/save/parameters
   ```

2. Use `MarkerDetect.py` to detect markers in images and analyze them:
   ```
   python MarkerDetect.py --image path/to/image --parameters path/to/camera/parameters
   ```

3. The calibration results and marker detection outputs will be saved to the specified output directory.

## Acknowledgments

This project is conducted at Nanyang Technological University, Singapore. We acknowledge the support and resources provided by NTU for this research.

## License

This project is licensed under the MIT License.