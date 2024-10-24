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
            
            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # Add label with confidence
            label_text = f'{label} {confidence:.2f}'
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            cv2.rectangle(frame, (xmin, ymin - label_size[1] - 10), (xmin + label_size[0], ymin), (0, 255, 0), -1)
            cv2.putText(frame, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # Show the frame
    cv2.imshow('Live Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()