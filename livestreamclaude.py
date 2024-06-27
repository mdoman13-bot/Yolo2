import cv2
from ultralytics import YOLO
import pafy
import time
from collections import defaultdict

# Initialize YOLOv8 model
model = YOLO('models/yolov8n.pt')  # or use 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt' for larger models


# Create a video object
video_path = "media/IMG_4583.MOV.mov"
cap = cv2.VideoCapture(video_path)


# Initialize object count
object_count = defaultdict(int)

# Set the interval for counting (e.g., every 5 seconds)
count_interval = 5
last_count_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Process the results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get the class of the detected object
            c = box.cls
            class_name = model.names[int(c)]
            
            # Increment the count for this class
            object_count[class_name] += 1

    # Check if it's time to print the count
    current_time = time.time()
    if current_time - last_count_time >= count_interval:
        print("\nDetected objects count:")
        for obj, count in object_count.items():
            print(f"{obj}: {count}")
        print("-------------------------")
        
        # Reset the count and update the last count time
        object_count = defaultdict(int)
        last_count_time = current_time

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()