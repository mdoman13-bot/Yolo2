import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

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

class StreamProcessor:
    def __init__(self, stream_url, model_path, device):
        self.stream_url = stream_url
        self.cap = cv2.VideoCapture(stream_url)
        self.model = YOLO(model_path).to(device)
        self.heatmap = None
        self.lock = Lock()
        self.device = device

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        results = self.model.predict(frame, verbose=False)
        
        with self.lock:
            if self.heatmap is None:
                self.heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
            
            for result in results:
                for box in result.boxes:
                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    label = self.model.names[class_id]
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
                    
                    label_text = f"{label} {confidence:.2f}"
                    font_scale = 0.5
                    cv2.putText(frame, label_text, (xmin, ymin - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 1)
                    
                    self.heatmap[ymin:ymax, xmin:xmax] += confidence
        
        return frame

    def cleanup(self):
        self.cap.release()

class MultiStreamDetector:
    def __init__(self, model_path='models/yolov8n.pt', grid_size=(4, 4)):
        self.grid_size = grid_size
        self.stream_processors = []
        
        # Distribute models across available GPUs if multiple are available
        num_gpus = torch.cuda.device_count()
        devices = [f'cuda:{i}' for i in range(num_gpus)] if num_gpus > 0 else ['cpu']
        
        for i, url in enumerate(STREAM_URLS):
            device = devices[i % len(devices)]
            processor = StreamProcessor(url, model_path, device)
            self.stream_processors.append(processor)
        
        self.executor = ThreadPoolExecutor(max_workers=len(STREAM_URLS))

    def process_streams(self):
        # Process all streams concurrently
        futures = [self.executor.submit(processor.process_frame) 
                  for processor in self.stream_processors]
        
        frames = []
        for future in futures:
            frame = future.result()
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frames.append(frame)
        
        return frames

    def run(self):
        cv2.namedWindow('Multi-stream Detection', cv2.WINDOW_NORMAL)
        
        try:
            while True:
                frames = self.process_streams()
                
                # Create grid display
                rows = []
                for i in range(0, len(frames), self.grid_size[1]):
                    row_frames = frames[i:i + self.grid_size[1]]
                    if len(row_frames) < self.grid_size[1]:
                        row_frames.extend([np.zeros_like(frames[0])] * (self.grid_size[1] - len(row_frames)))
                    rows.append(np.hstack(row_frames))
                
                grid_display = np.vstack(rows[:self.grid_size[0]])
                
                cv2.imshow('Multi-stream Detection', grid_display)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            self.cleanup()

    def cleanup(self):
        # Combine heatmaps from all processors
        combined_heatmap = None
        for processor in self.stream_processors:
            if processor.heatmap is not None:
                if combined_heatmap is None:
                    combined_heatmap = processor.heatmap.copy()
                else:
                    combined_heatmap += cv2.resize(processor.heatmap, 
                                                 (combined_heatmap.shape[1], 
                                                  combined_heatmap.shape[0]))
            processor.cleanup()
        
        self.executor.shutdown()
        cv2.destroyAllWindows()
        
        # Display final heatmap
        if combined_heatmap is not None:
            heatmap = cv2.normalize(combined_heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap = np.uint8(heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            plt.imshow(heatmap)
            plt.title('Combined Object Detection Heatmap')
            plt.axis('off')
            plt.show()

if __name__ == "__main__":
    detector = MultiStreamDetector()
    detector.run()