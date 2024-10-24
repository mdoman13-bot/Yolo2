import cv2
import torch
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8s.pt')  # Replace with the path to your model
# To get this to work, I had to pip uninstall opencv-python then pip install opencv-python
# Open the live stream
stream_url = "https://skysfs3.trafficwise.org/rtplive/INDOT_257_IlQ0iAJPF3zCjVhF/playlist.m3u8"
cap = cv2.VideoCapture(stream_url)

# Create a window
cv2.namedWindow('Live Detection', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model.predict(frame)

    # Display the results
    for result in results:
        for box in result.boxes:
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = box.conf[0]

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

    # Show the frame
    cv2.imshow('Live Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()