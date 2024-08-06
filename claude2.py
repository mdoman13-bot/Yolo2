import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict
import matplotlib.pyplot as plt

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
class_counts = defaultdict(int)
class_counts_over_time = defaultdict(lambda: defaultdict(int))
processed_frames = 0

while True:
    # Read a new frame from the video
    ret, frame = video.read()
    if not ret:
        break  # Break the loop if there are no frames left

    frame_count += 1

    if frame_count % 30 == 0:  # Process every 30th frame
        processed_frames += 1
        
        # Process the frame
        detections = sv.InferenceSlicer(callback=callback)(frame)
        
        # Update class counts
        for class_id in detections.class_id:
            class_name = model.names[class_id]
            class_counts[class_name] += 1
            class_counts_over_time[class_name][processed_frames] = class_counts[class_name]

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
        
        # Reset the class counts for the current frame
        class_counts.clear()

        # Every 5 processed frames (150 actual frames), plot the graph
        if processed_frames % 5 == 0:
            plt.figure(figsize=(12, 6))
            
            # Get the 8 most detected items
            top_8_classes = sorted(class_counts_over_time.items(), 
                                   key=lambda x: max(x[1].values()), 
                                   reverse=True)[:8]
            
            for class_name, counts in top_8_classes:
                frames = list(counts.keys())
                counts = list(counts.values())
                plt.plot(frames, counts, label=class_name)
            
            plt.xlabel('Processed Frames')
            plt.ylabel('Cumulative Detections')
            plt.title('Top 8 Detected Objects Over Time')
            plt.legend()
            plt.grid(True)
            
            # Save the plot as an image
            plt.savefig(f'detection_graph_{processed_frames}.png')
            plt.close()
            
            print(f"Graph saved as detection_graph_{processed_frames}.png")

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release the video capture object and close all windows
video.release()
cv2.destroyAllWindows()