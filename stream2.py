import cv2
import subprocess
import numpy as np
from ultralytics import YOLO
from threading import Thread
from queue import Queue
import time
import os
import sys
import csv
import requests
from datetime import datetime
import re

def analyze_ts_pattern(base_url, num_samples=5):
    """
    Analyze the TS file naming pattern and save to CSV
    """
    pattern_data = []
    current_ts = "media_w741173679_3.ts"  # Starting point
    
    print("Analyzing TS file naming pattern...")
    try:
        # Extract components from known TS file
        parts = re.match(r'media_w(\d+)_(\d+)\.ts', current_ts)
        if parts:
            base_number = parts.group(1)
            sequence = parts.group(2)
            
            # Try a few variations to understand the pattern
            for i in range(num_samples):
                test_url = f"{base_url}/media_w{base_number}_{i}.ts"
                response = requests.head(test_url)
                pattern_data.append({
                    'base_number': base_number,
                    'sequence': i,
                    'full_name': f"media_w{base_number}_{i}.ts",
                    'exists': response.status_code == 200
                })
                
        # Save pattern analysis to CSV
        csv_path = 'ts_pattern_analysis.csv'
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['base_number', 'sequence', 'full_name', 'exists']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(pattern_data)
            
        print(f"Pattern analysis saved to {csv_path}")
        return pattern_data
            
    except Exception as e:
        print(f"Error analyzing TS pattern: {e}")
        return None

class StreamProcessor:
    def __init__(self, stream_url, ffmpeg_path=None, save_frames=False, show_detections=True):
        """
        Initialize the stream processor
        stream_url: Direct URL to the TS stream
        ffmpeg_path: Optional path to ffmpeg executable
        save_frames: Whether to save processed frames to disk
        show_detections: Whether to show detection boxes on frames
        """
        self.stream_url = stream_url
        self.width = 352
        self.height = 240
        self.frame_queue = Queue(maxsize=30)
        self.processed_queue = Queue(maxsize=30)
        self.running = False
        self.ffmpeg_path = ffmpeg_path or self.find_ffmpeg()
        self.save_frames = save_frames
        self.show_detections = show_detections
        
        # Create frames directory if saving frames
        if self.save_frames:
            self.frames_dir = 'frames'
            os.makedirs(self.frames_dir, exist_ok=True)
            print(f"Frames will be saved to {self.frames_dir}/")
        
        if not self.ffmpeg_path:
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg and add it to PATH or provide path to ffmpeg executable."
            )
        
        # Initialize YOLO model
        print("Loading YOLO model...")
        self.model = YOLO('yolov8n.pt')
        
    def find_ffmpeg(self):
        """Find ffmpeg executable in system PATH or common locations"""
        if sys.platform.startswith('win'):
            common_paths = [
                r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
                r"C:\ffmpeg\bin\ffmpeg.exe",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg.exe")
            ]
            
            if os.system("where ffmpeg > nul 2>&1") == 0:
                return "ffmpeg"
                
            for path in common_paths:
                if os.path.isfile(path):
                    return path
                    
            return None
        else:
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
            
            print(f"Starting stream capture...")
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
            
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
        frame_count = 0
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                
                try:
                    # Run YOLO detection
                    results = self.model(frame)
                    
                    # Choose whether to show detection boxes
                    if self.show_detections:
                        processed_frame = results[0].plot()
                    else:
                        processed_frame = frame.copy()
                    
                    # Save frame if enabled
                    if self.save_frames:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = os.path.join(self.frames_dir, f"frame_{timestamp}.jpg")
                        cv2.imwrite(filename, processed_frame)
                    
                    if not self.processed_queue.full():
                        self.processed_queue.put(processed_frame)
                        
                    frame_count += 1
                    if frame_count % 30 == 0:  # Print stats every 30 frames
                        print(f"Processed {frame_count} frames")
                        
                except Exception as e:
                    print(f"Error processing frame: {e}")
            else:
                time.sleep(0.001)

    def display_frames(self):
        """Display processed frames"""
        while self.running:
            if not self.processed_queue.empty():
                frame = self.processed_queue.get()
                
                # Create a resized display window (2x larger for better visibility)
                display_frame = cv2.resize(frame, (self.width * 2, self.height * 2))
                
                try:
                    cv2.imshow('Stream', display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):  # Quit
                        self.running = False
                    elif key == ord('s'):  # Toggle show detections
                        self.show_detections = not self.show_detections
                        print(f"Detection visualization: {'ON' if self.show_detections else 'OFF'}")
                    elif key == ord('f'):  # Toggle frame saving
                        self.save_frames = not self.save_frames
                        if self.save_frames:
                            os.makedirs(self.frames_dir, exist_ok=True)
                        print(f"Frame saving: {'ON' if self.save_frames else 'OFF'}")
                except Exception as e:
                    print(f"Display error: {e}")
            else:
                time.sleep(0.001)

    def run(self):
        """Main processing loop"""
        self.running = True
        
        print("\nControls:")
        print("  q: Quit")
        print("  s: Toggle detection visualization")
        print("  f: Toggle frame saving")
        print("\nStarting processing...")
        
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
    STREAM_URL = "https://skysfs4.trafficwise.org/rtplive/INDOT_262__7ypTvHKbwMpXYJD/media_w1021746165_105.ts"
    BASE_URL = "https://public.carsprogram.org/cameras/IN/INDOT_262__7ypTvHKbwMpXYJD"
    
    # Analyze TS naming pattern
    analyze_ts_pattern(BASE_URL)
    
    try:
        # Initialize with frame saving disabled by default, detection visualization enabled
        processor = StreamProcessor(STREAM_URL, save_frames=False, show_detections=True)
        processor.run()
    except Exception as e:
        print(f"Error: {e}")