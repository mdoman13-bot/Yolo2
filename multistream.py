import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

# List of stream URLs
STREAM_URLS = [
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

class MultiStreamDetector:
    def __init__(self, model_path='models/yolov8n.pt', grid_size=(4, 4)):
        self.model = YOLO(model_path)
        self.grid_size = grid_size
        self.streams = []
        self.heatmaps = []
        self.executor = ThreadPoolExecutor(max_workers=len(STREAM_URLS))
        
        # Enable CUDA if available
        if torch.cuda.is_available():
            self.model.to('cuda')
        
        # Initialize video captures
        for url in STREAM_URLS:
            cap = cv2.VideoCapture(url)
            self.streams.append(cap)
            self.heatmaps.append(None)

    def read_stream(self, stream_idx):
        ret, frame = self.streams[stream_idx].read()
        return stream_idx, ret, frame

    def process_frame(self, frame):
        if frame is None:
            return None, None
        
        # Batch process frames with YOLO
        results = self.model.predict(frame, verbose=False)
        
        # Draw detections
        for result in results:
            for box in result.boxes:
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                label = self.model.names[class_id]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
                
                # Add label with confidence
                label_text = f"{label} {confidence:.2f}"
                font_scale = 0.5
                cv2.putText(frame, label_text, (xmin, ymin - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 1)
                
                # Update heatmap
                if self.heatmaps[0] is None:
                    self.heatmaps[0] = np.zeros_like(frame[:, :, 0], dtype=np.float32)
                self.heatmaps[0][ymin:ymax, xmin:xmax] += confidence
        
        return frame, results

    def run(self):
        cv2.namedWindow('Multi-stream Detection', cv2.WINDOW_NORMAL)
        
        try:
            while True:
                # Read frames concurrently
                future_frames = list(self.executor.map(self.read_stream, range(len(self.streams))))
                
                # Process valid frames
                frames = []
                for stream_idx, ret, frame in future_frames:
                    if ret:
                        processed_frame, _ = self.process_frame(frame)
                        frames.append(processed_frame)
                    else:
                        frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
                
                if not frames:
                    break
                
                # Create grid display
                rows = []
                for i in range(0, len(frames), self.grid_size[1]):
                    row_frames = frames[i:i + self.grid_size[1]]
                    if len(row_frames) < self.grid_size[1]:
                        row_frames.extend([np.zeros_like(frames[0])] * (self.grid_size[1] - len(row_frames)))
                    rows.append(np.hstack(row_frames))
                
                grid_display = np.vstack(rows[:self.grid_size[0]])
                
                # Show the combined frame
                cv2.imshow('Multi-stream Detection', grid_display)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            self.cleanup()

    def cleanup(self):
        for stream in self.streams:
            stream.release()
        cv2.destroyAllWindows()
        self.executor.shutdown()
        
        # Display final heatmap
        if self.heatmaps[0] is not None:
            heatmap = cv2.normalize(self.heatmaps[0], None, 0, 255, cv2.NORM_MINMAX)
            heatmap = np.uint8(heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            plt.imshow(heatmap)
            plt.title('Combined Object Detection Heatmap')
            plt.axis('off')
            plt.show()

if __name__ == "__main__":
    detector = MultiStreamDetector()
    detector.run()