import cv2
import subprocess
import numpy as np
from ultralytics import YOLO
from threading import Thread
from queue import Queue
import time

class StreamProcessor:
    def __init__(self, stream_url):
        """
        Initialize the stream processor
        stream_url: Direct URL to the TS stream
        """
        self.stream_url = stream_url
        self.width = 352
        self.height = 240
        self.frame_queue = Queue(maxsize=30)
        self.processed_queue = Queue(maxsize=30)
        self.running = False
        
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')
        
    def capture_stream(self):
        """Capture frames directly from the stream using ffmpeg"""
        try:
            command = [
                'ffmpeg',
                '-i', self.stream_url,
                '-f', 'image2pipe',
                '-pix_fmt', 'bgr24',
                '-vcodec', 'rawvideo',
                '-reconnect', '1',
                '-reconnect_streamed', '1',
                '-reconnect_delay_max', '5',
                '-hide_banner',
                '-loglevel', 'error',
                '-'
            ]
            
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
            
            # Calculate frame size for 352x240
            frame_size = self.width * self.height * 3
            
            while self.running:
                raw_frame = process.stdout.read(frame_size)
                if not raw_frame:
                    print("Stream ended or error occurred. Restarting...")
                    process.terminate()
                    time.sleep(1)
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
                    continue
                
                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                try:
                    frame = frame.reshape((self.height, self.width, 3))
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)
                except ValueError as e:
                    print(f"Frame reshape error: {e}")
                    continue
                    
        except Exception as e:
            print(f"Stream capture error: {e}")
            self.running = False

    def process_frames(self):
        """Process frames using YOLO"""
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                
                # Run YOLO detection
                results = self.model(frame)
                
                # Draw results on frame
                annotated_frame = results[0].plot()
                
                if not self.processed_queue.full():
                    self.processed_queue.put(annotated_frame)
            else:
                time.sleep(0.001)  # Small sleep when queue is empty

    def display_frames(self):
        """Display processed frames"""
        while self.running:
            if not self.processed_queue.empty():
                frame = self.processed_queue.get()
                
                # Create a resized display window (2x larger for better visibility)
                display_frame = cv2.resize(frame, (self.width * 2, self.height * 2))
                cv2.imshow('Stream', display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
            else:
                time.sleep(0.001)  # Small sleep when queue is empty

    def run(self):
        """Main processing loop"""
        self.running = True
        
        # Start all processing threads
        capture_thread = Thread(target=self.capture_stream)
        process_thread = Thread(target=self.process_frames)
        display_thread = Thread(target=self.display_frames)
        
        capture_thread.start()
        process_thread.start()
        display_thread.start()
        
        try:
            # Wait for threads to complete
            capture_thread.join()
            process_thread.join()
            display_thread.join()
        except KeyboardInterrupt:
            self.running = False
            print("\nShutting down gracefully...")
        
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    STREAM_URL = "https://skysfs4.trafficwise.org/rtplive/INDOT_262__7ypTvHKbwMpXYJD/media_w741173679_3.ts"
    processor = StreamProcessor(STREAM_URL)
    processor.run()