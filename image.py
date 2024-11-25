from ultralytics import YOLO

# Initialize a YOLO-World model
model = YOLO("models/yolov8s-world.pt")  # or choose yolov8m/l-world.pt

# Define custom classes
model.set_classes(["person", "pool", "water"])

# Execute prediction for specified categories on an image
results = model.predict("media/beach2.jpg")

# Show results
results[0].show()