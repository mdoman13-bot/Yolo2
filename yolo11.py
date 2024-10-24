from ultralytics import YOLO
import cv2
# image = cv2.imread('media/beach.png')
# cv2.imshow('Beach Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# Load a model
model = YOLO("models/yolo11n.pt")  # load an official detection model
# model = YOLO("yolo11n-seg.pt")  # load an official segmentation model
# model = YOLO("path/to/best.pt")  # load a custom model

# Track with the model

results = model.track(source="https://skysfs3.trafficwise.org/rtplive/INDOT_257_IlQ0iAJPF3zCjVhF/playlist.m3u8", show=True, stream="True")
# results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")