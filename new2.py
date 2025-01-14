import cv2
import torch
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

# Load the YOLO model
model = YOLO('models/yolov8n.pt')  # Replace with the path to your model
model.to('cuda')  # Use GPU for inference
# List of stream URLs
stream_urls = [
    "https://skysfs3.trafficwise.org/rtplive/INDOT_257_IlQ0iAJPF3zCjVhF/playlist.m3u8",
    "https://skysfs4.trafficwise.org/rtplive/INDOT_262__7ypTvHKbwMpXYJD/playlist.m3u8",
    "https://skysfs3.trafficwise.org/rtplive/INDOT_703_I3RqqDqcbqI1A_Z3/playlist.m3u8",
    "https://skysfs3.trafficwise.org/rtplive/INDOT_261_B6pE8gVw3RJ7YdXn/playlist.m3u8",
    "https://skysfs4.trafficwise.org/rtplive/INDOT_260_l6kGZjfYqqL9vgE9/playlist.m3u8",
    "https://skysfs3.trafficwise.org/rtplive/INDOT_257_IlQ0iAJPF3zCjVhF/playlist.m3u8",
    "https://skysfs4.trafficwise.org/rtplive/INDOT_32_8Aa6tXj3dYK35_P2/playlist.m3u8",
    "https://skysfs3.trafficwise.org/rtplive/INDOT_579_6c5KpTSKIJWts7HU/playlist.m3u8",
    "https://skysfs3.trafficwise.org/rtplive/INDOT_591_ikD7yYJFi8SlJxfy/playlist.m3u8"
]

# Create video captures for each stream
caps = [cv2.VideoCapture(url) for url in stream_urls]

# Create a window
cv2.namedWindow('Live Detection', cv2.WINDOW_NORMAL)

heatmap = None
detection = False  # Set this to False to disable YOLO inference

while True:
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            frames.append(np.zeros((480, 640, 3), dtype=np.uint8))  # Placeholder for empty frames
        else:
            frames.append(frame)

    # Combine frames into a grid
    grid_size = 4
    frame_height, frame_width = frames[0].shape[:2]
    grid_frame = np.zeros((frame_height * grid_size, frame_width * grid_size, 3), dtype=np.uint8)

    for i, frame in enumerate(frames):
        row = i // grid_size
        col = i % grid_size
        grid_frame[row * frame_height:(row + 1) * frame_height, col * frame_width:(col + 1) * frame_width] = frame

    if detection:
        # Perform object detection on the combined frame
        results = model.predict(grid_frame)

        # Initialize heatmap
        if heatmap is None:
            heatmap = np.zeros((grid_frame.shape[0], grid_frame.shape[1]), dtype=np.float32)

        # Display the results
        for result in results:
            for box in result.boxes:
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])  # Convert tensor to scalar

                # Get class name if available, otherwise use class ID
                label = model.names[class_id] if hasattr(model, 'names') else str(class_id)
                
                # Draw bounding box with smaller thickness
                cv2.rectangle(grid_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
                
                # Draw label with smaller font scale
                font_scale = 0.5
                font_thickness = 1
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                label_x = xmin
                label_y = ymin - 10 if ymin - 10 > 10 else ymin + 10
                cv2.putText(grid_frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)

                # Update heatmap
                heatmap[ymin:ymax, xmin:xmax] += confidence

    # Show the combined frame
    cv2.imshow('Live Detection', grid_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the captures and destroy all windows
for cap in caps:
    cap.release()
cv2.destroyAllWindows()

if detection:
    # Normalize heatmap
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = np.uint8(heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Display heatmap using matplotlib
    plt.imshow(heatmap)
    plt.title('Object Detection Heatmap')
    plt.axis('off')
    plt.show()
