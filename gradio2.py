import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict
import gradio as gr
import os

def process_video(model_path="models/yolov8n.pt", video_path="media/silas1.mp4"):
    # Load the YOLO model
    model = YOLO(model_path)

    # Define the callback function for processing
    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = model(image_slice)[0]
        return sv.Detections.from_ultralytics(result)

    # Initialize the annotators
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Open the video file
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    class_counts = defaultdict(int)
    global_class_counts = {}
    
    frames = []

    while True:
        # Read a new frame from the video
        ret, frame = video.read()
        if not ret:
            break  # Break the loop if there are no frames left

        frame_count += 1
        if frame_count % 15 == 0:  # Process every 15th frame
            # Process the frame
            detections = sv.InferenceSlicer(callback=callback)(frame)

            # Update class counts
            for class_id in detections.class_id:
                class_counts[model.names[class_id]] += 1

            annotated_frame = bounding_box_annotator.annotate(
                scene=frame, detections=detections)
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections)

            # Convert BGR to RGB
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frames.append(annotated_frame_rgb)

            # Update global counters
            for class_name, count in class_counts.items():
                if class_name in global_class_counts:
                    global_class_counts[class_name] += count
                else:
                    global_class_counts[class_name] = count

            # Reset the class counts
            class_counts.clear()

    # Release the video capture object
    video.release()

    return frames, global_class_counts

def gradio_interface(model_path, video_path):
    frames, global_counts = process_video(model_path, video_path)
    
    # Convert global counts to a string
    counts_str = "\n".join([f"{k}: {v}" for k, v in global_counts.items()])
    
    return frames, counts_str

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Model Path", value="models/yolov8n.pt"),
        gr.Textbox(label="Video Path", value="media/silas1.mp4")
    ],
    outputs=[
        gr.Gallery(label="Processed Frames"),
        gr.Textbox(label="Global Class Counts")
    ],
    title="YOLO Video Detection",
    description="Process a video using YOLO object detection and display results."
)

iface.launch()