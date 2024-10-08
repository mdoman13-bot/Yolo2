import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict

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
video = cv2.VideoCapture("media/silas1.mp4")

frame_count = 0
class_counts = defaultdict(int)
global_class_counts = {}

while True:
    # Read a new frame from the video
    ret, frame = video.read()
    if not ret:
        break  # Break the loop if there are no frames left

    frame_count += 1

    if frame_count % 15 == 0:  # Process every 30th frame
        # Process the frame
        detections = sv.InferenceSlicer(callback=callback)(frame)
        
        # Update class counts
        for class_id in detections.class_id:
            class_counts[model.names[class_id]] += 1
    
        annotated_frame = bounding_box_annotator.annotate(
            scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections)
    
        # Display the annotated frame
        cv2.imshow("Annotated Frame", annotated_frame)
    
        # Print the number of detections for each class
        print(f"Frame {frame_count}:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")
            

        # Step 2: Update global counters
        for class_name, count in class_counts.items():
            if class_name in global_class_counts:
                global_class_counts[class_name] += count
            else:
                global_class_counts[class_name] = count
        
        # Reset the class counts
        class_counts.clear()
    

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break
print("Global class counts:\n", global_class_counts.items())
# Release the video capture object and close all windows
video.release()
cv2.destroyAllWindows()