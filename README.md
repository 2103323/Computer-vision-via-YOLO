# Computer-vision-via-YOLO

Uploading DEMO YOLO VIDEO.mp4…

# YOLO Object Detection with OpenCV

This repository contains Python code for real-time object detection using the YOLO (You Only Look Once) model with OpenCV. The code utilizes the Ultralytics library for implementing the YOLO model and PyTorch for deep learning functionalities.

## Requirements

To run the code, you need to have the following installed:

- Python 3.x
- OpenCV
- PyTorch
- Ultralytics library

You can install the required libraries using pip:

```bash
pip install opencv-python torch torchvision
pip install 'git+https://github.com/ultralytics/yolov5.git'
```

## Usage

1. Clone this repository:

```bash
git clone https://github.com/2103323/Computer-vision-via-YOLO
cd yolo-opencv
```

2. Download the pre-trained YOLO weights (`yolov8x.pt`) and place it in the project directory.

3. Replace `"people.mp4"` with the path to your input video file.

4. Run the Python script:

```bash
python yolo_opencv.py
```

5. Press the 'Esc' key to exit the video stream and close the OpenCV windows.

## How it works

1. The YOLO model is loaded using the Ultralytics library.

2. The input video file is read frame by frame using OpenCV.

3. Each frame is processed by the YOLO model to detect objects.

4. Bounding boxes are drawn around the detected objects on the frame.

5. The frame with bounding boxes is displayed using OpenCV.

6. The process continues until the 'Esc' key is pressed.

## Notes

- The code checks if Multi-Processing Service (MPS) is available in PyTorch. If available, YOLO model inference is performed using MPS for faster processing.

- You can modify the code to use a different pre-trained YOLO model or adjust the confidence threshold for object detection.

- Ensure that the input video file exists and is accessible.

- The code is intended for educational purposes and can be extended for various object detection applications.
