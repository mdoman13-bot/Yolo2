import cv2
import torch
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

# Load the YOLO model
model = YOLO('models/yolov10n.pt')  # Replace with the path to your model

# List of stream URLs
stream_urls = [
    "https://skysfs3.trafficwise.org/rtplive/INDOT_257_IlQ0iAJPF3zCjVhF/playlist.m3u8",
    "https://skysfs4.trafficwise.org/rtplive/INDOT_262__7ypTvHKbwMpXYJD/playlist.m3u8",
    "https://skysfs3.trafficwise.org/rtplive/INDOT_703_I3RqqDqcbqI1A_Z3/playlist.m3u8",
    "https://skysfs3.trafficwise.org/rtplive/INDOT_261_B6pE8gVw3RJ7YdXn/playlist.m3u8"
]

# Initialize video captures
caps = [cv2.VideoCapture(url) for url in stream_urls]

# Create a window
cv2.namedWindow('Live Detection', cv2.WINDOW_NORMAL)

# Function to draw text on frame
def draw_text(frame, text, position, color=(0, 255, 0), font_scale=0.5, thickness=1):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

# Function to resize frame to a common size
def resize_frame(frame, size=(640, 480)):
    return cv2.resize(frame, size)

while True:
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Create a black frame
            draw_text(frame, 'Loading Failed', (50, 50), color=(0, 0, 255))
        else:
            # Perform object detection
            results = model.predict(frame)

            # Display the results
            for result in results:
                for box in result.boxes:
                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])  # Convert tensor to scalar

                    # Get class name if available, otherwise use class ID
                    label = model.names[class_id] if hasattr(model, 'names') else str(class_id)
                    
                    # Draw bounding box with smaller thickness
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
                    
                    # Draw label with smaller font scale
                    font_scale = 0.5
                    font_thickness = 1
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    label_x = xmin
                    label_y = ymin - 10 if ymin - 10 > 10 else ymin + 10
                    cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)

            draw_text(frame, 'Stream Loading', (50, 50))

        # Resize frame to a common size
        frame = resize_frame(frame)
        frames.append(frame)

    # Combine frames into a grid
    top_row = np.hstack(frames[:2])
    bottom_row = np.hstack(frames[2:])
    combined_frame = np.vstack((top_row, bottom_row))

    # Show the combined frame
    cv2.imshow('Live Detection', combined_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the captures and destroy all windows
for cap in caps:
    cap.release()
cv2.destroyAllWindows()