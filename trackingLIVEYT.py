import cv2
from ultralytics import YOLO

# Configure the tracking parameters and run the tracker
model = YOLO('models/yolov8n.pt')
results = model.track(source="media/newvid2.mp4", conf=0.3, iou=0.5, show=True)

# Add functionality to end the program when 'q' is pressed
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

