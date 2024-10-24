import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the YOLO model
model = YOLO('yolov8s.pt')  # Replace with the path to your model

# Open the live stream
stream_url = "https://skysfs3.trafficwise.org/rtplive/INDOT_257_IlQ0iAJPF3zCjVhF/playlist.m3u8"
cap = cv2.VideoCapture(stream_url)

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
            label = box.cls
            confidence = box.conf

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # Assuming `confidence` is a tensor, convert it to a float
            confidence = confidence.item() if isinstance(confidence, torch.Tensor) else confidence
            cv2.putText(frame, f'{label} {confidence:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Use matplotlib to display the frame
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    plt.pause(0.001)  # Pause to allow the plot to update
    plt.clf()  # Clear the plot for the next frame

cap.release()
cv2.destroyAllWindows()