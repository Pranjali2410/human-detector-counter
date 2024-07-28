# Human Detector and Counter

A Python-based real-time human detection and counting system utilizing YOLOv3 and OpenCV. This project identifies and counts humans in video feeds, making it ideal for security and monitoring applications.

## Features
- Real-time human detection and counting
- Utilizes YOLOv3 for accurate object detection
- Provides real-time feedback with bounding boxes and counts

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/human-detector-counter.git
   
2. Navigate to the project directory:
   ```bash
   cd human-detector-counter

3. Install dependencies:
   ```bash
   pip install opencv-python numpy

4. Download YOLOv3 Files:
Download the YOLOv3 weights, configuration (cfg), and class names files from the YOLO website:
  - yolov3.weights
  - yolov3.cfg
  - coco.names
Place these files in the project directory.

## Usage
To run the human detection and counting system:

1. Run the Python Script:
   python human_detector.py

2. Interact with the Application:
  - The script will open a video capture window displaying the live feed with detected humans highlighted by bounding boxes.
  - The total count of detected humans will be displayed on the screen.
  - Press q to quit the application.

## Expected results
  - Real-Time Detection: The script will identify and count humans in the video feed.
  - Bounding Boxes: You'll see green boxes around detected people.
  - Human Count: The number of detected humans will be shown at the top-left of the video feed.

## Project Report
A detailed report on the development and results of the Human Detector and Counter project is available.
