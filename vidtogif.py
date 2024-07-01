import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("models/yolov8n.pt")

# Define the callback function for processing
def callback(image_slice: np.ndarray) -> sv.Detections:
    result = model(image_slice)[0]
    return sv.Detections.from_ultralytics(result)

# Initialize the annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Open the video file
video = cv2.VideoCapture("media/drone_cars.mp4")

frame_count = 0
while True:
    # Read a new frame from the video
    ret, frame = video.read()
    if not ret:
        break  # Break the loop if there are no frames left

    frame_count += 1
    if frame_count % 30 == 0:  # Check if it is the 30th frame
        # Process the frame
        detections = sv.InferenceSlicer(callback=callback)(frame)
        annotated_frame = bounding_box_annotator.annotate(
            scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections)

        # Display the annotated frame
        cv2.imshow("Annotated Frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

# Release the video capture object and close all windows
video.release()
cv2.destroyAllWindows()
