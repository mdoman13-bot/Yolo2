import cv2
import subprocess
import numpy as np
from ultralytics import YOLO
from threading import Thread
from queue import Queue
import time
import os
import sys

class StreamProcessor:
    def __init__(self, stream_url, ffmpeg_path=None):
        """
        Initialize the stream processor
        stream_url: Direct URL to the TS stream
        ffmpeg_path: Optional path to ffmpeg executable
        """
        self.stream_url = stream_url
        self.width = 352
        self.height = 240
        self.frame_queue = Queue(maxsize=30)
        self.processed_queue = Queue(maxsize=30)
        self.running = False
        self.ffmpeg_path = ffmpeg_path or self.find_ffmpeg()
        
        if not self.ffmpeg_path:
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg and add it to PATH or provide path to ffmpeg executable."
            )
        
        # Initialize YOLO model
        print("Loading YOLO model...")
        self.model = YOLO('yolov8n.pt')
        
    def find_ffmpeg(self):
        """Find ffmpeg executable in system PATH or common locations"""
        # Check if ffmpeg is in PATH
        if sys.platform.startswith('win'):
            # Common Windows installation locations
            common_paths = [
                r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
                r"C:\ffmpeg\bin\ffmpeg.exe",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg.exe")
            ]
            
            # First check PATH
            if os.system("where ffmpeg > nul 2>&1") == 0:
                return "ffmpeg"
                
            # Then check common locations
            for path in common_paths:
                if os.path.isfile(path):
                    return path
                    
            return None
        else:
            # Unix-like systems
            return "ffmpeg" if os.system("which ffmpeg > /dev/null 2>&1") == 0 else None

    def capture_stream(self):
        """Capture frames directly from the stream using ffmpeg"""
        try:
            command = [
                self.ffmpeg_path,
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
            
            print(f"Starting ffmpeg with command: {' '.join(command)}")
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
            _, stderr = process.communicate()
            print(f"FFmpeg error output: {stderr.decode()}")
            self.running = False

    def process_frames(self):
        """Process frames using YOLO"""
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                
                try:
                    # Run YOLO detection
                    results = self.model(frame)
                    
                    # Draw results on frame
                    annotated_frame = results[0].plot()
                    
                    if not self.processed_queue.full():
                        self.processed_queue.put(annotated_frame)
                except Exception as e:
                    print(f"Error processing frame: {e}")
            else:
                time.sleep(0.001)

    def save_frame(self, frame, filename):
        """Save a frame to disk"""
        try:
            cv2.imwrite(filename, frame)
            print(f"Saved frame to {filename}")
        except Exception as e:
            print(f"Error saving frame: {e}")

    def display_frames(self):
        """Display processed frames and save periodically"""
        frame_count = 0
        
        while self.running:
            if not self.processed_queue.empty():
                frame = self.processed_queue.get()
                
                # Create a resized display window (2x larger for better visibility)
                display_frame = cv2.resize(frame, (self.width * 2, self.height * 2))
                
                try:
                    cv2.imshow('Stream', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                except Exception as e:
                    print(f"Display error (falling back to saving frames): {e}")
                    # Save every 30th frame as fallback
                    if frame_count % 30 == 0:
                        self.save_frame(display_frame, f"frame_{frame_count}.jpg")
                    frame_count += 1
            else:
                time.sleep(0.001)

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
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"Error closing windows: {e}")

# Example usage
if __name__ == "__main__":
    STREAM_URL = "https://skysfs4.trafficwise.org/rtplive/INDOT_262__7ypTvHKbwMpXYJD/media_w741173679_3.ts"
    
    # You can specify the path to ffmpeg.exe if it's not in PATH
    # FFMPEG_PATH = r"C:\path\to\ffmpeg.exe"
    # processor = StreamProcessor(STREAM_URL, FFMPEG_PATH)
    
    try:
        processor = StreamProcessor(STREAM_URL)
        processor.run()
    except Exception as e:
        print(f"Error: {e}")