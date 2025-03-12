# CV Viaduct Project

This project involves detecting markers and calibrating an iPhone camera(iPhone 13 Pro Max Main Wide Camera) using OpenCV and ArUco markers. It includes scripts for processing images and calculating camera parameters.

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

1. Run `iphoneCameraCali.py` to calibrate the camera.
2. Use `MarkerDetect.py` to detect markers in images and analyze them.

## License

This project is licensed under the MIT License. 