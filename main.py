import cv2  # Importing OpenCV library for image processing
from ultralytics import YOLO  # Importing YOLO model from Ultralytics library
import torch  # Importing PyTorch library

# Loading YOLO model
model = YOLO("yolov8x.pt")

# Opening the video file
capture = cv2.VideoCapture("people.mp4")

# Looping through each frame of the video
while True:
    # Reading the frame from the video
    is_frame, frame = capture.read()
    
    # If there are no more frames to read, exit the loop
    if not is_frame:
        break
    
    # Checking if MPS (Multi-Processing Service) is available in PyTorch
    if torch.backends.mps.is_available():
        # If available, running YOLO model using MPS
        results = model(frame, devices="mps")
    else:
        # If MPS is not available, running YOLO model normally
        results = model(frame)
    
    # Iterating through the detected objects in the frame
    for result in results:
        # Extracting bounding box coordinates for each detected object
        bboxes = result.boxes.xyxy.cpu().numpy().astype("int")
        
        # Drawing bounding boxes around detected objects
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Displaying the frame with bounding boxes
    cv2.imshow("Video", frame)
    
    # Waiting for user input (waitKey returns the ASCII value of the key pressed)
    key = cv2.waitKey(1)
    
    # If 'Esc' key is pressed, exit the loop
    if key == 27:
        break

# Releasing the video capture object
capture.release()

# Closing all OpenCV windows
cv2.destroyAllWindows()
