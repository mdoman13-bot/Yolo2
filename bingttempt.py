import cv2
import numpy as np
from ultralytics import YOLO
import imageio

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Function to process each frame
def process_frame(frame: np.ndarray):
    # Perform inference
    result = model(frame)[0]
    # Convert results to detections
    detections = sv.Detections.from_ultralytics(result)
    # Annotate frame
    annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
    return annotated_frame

# Initialize video capture and GIF writer
cap = cv2.VideoCapture('media/drone_cars.mp4')
gif_frames = []

# Process video
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % 30 == 0:
        processed_frame = process_frame(frame)
        gif_frames.append(processed_frame)
    frame_count += 1

# Release video capture
cap.release()

# Save frames as GIF
imageio.mimsave('labeled_frames.gif', gif_frames, fps=10)
