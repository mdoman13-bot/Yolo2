import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("models/yolov8s-worldv2.pt")
# model.to('mps')
# Define custom classes
model.set_classes(["pool", "car", "person", "road", "tree", "building", "sky"])

# Define the callback function for processing
def callback(image_slice: np.ndarray) -> sv.Detections:
    result = model(image_slice)[0]
    return sv.Detections.from_ultralytics(result)

# Initialize the annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Read the image
image = cv2.imread("media/pool1.png")

# Process the image
detections = sv.InferenceSlicer(callback=callback)(image)
annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

# Save or display the annotated image
cv2.imwrite("media/annotated_pool1.png", annotated_image)
cv2.imshow("Annotated Image", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()