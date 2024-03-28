# Import the required libraries
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8n model
model = YOLO("models/yolov8n.pt")

# Define the classes to be detected
classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Loop over the frames from the webcam
while True:
    # Read the current frame
    ret, frame = cap.read()

    # Check if the frame is valid
    if not ret:
        break

    # Resize the frame to 640x480
    frame = cv2.resize(frame, (640, 480))

    # Convert the frame to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Predict the objects in the frame using the model
    results = model(frame)

    # Loop over the detected objects
    for result in results:
        # Get the class index, confidence, and bounding box coordinates
        class_id, conf, x1, y1, x2, y2 = result

        # Get the class name
        class_name = classes[class_id]

        # Draw a rectangle around the object
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw a label with the class name and confidence
        cv2.putText(frame, f"{class_name}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Convert the frame back to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Show the frame on the screen
    cv2.imshow("Webcam", frame)

    # Wait for a key press
    key = cv2.waitKey(1)

    # If the key is ESC, break the loop
    if key == 27:
        break

# Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()
