import m3u8
import requests
import cv2
import numpy as np
from urllib.parse import urljoin
import time
from threading import Thread
from queue import Queue
import subprocess

class HLSStreamCapture:
    def __init__(self, base_url, playlist_url=None, ffmpeg_path='ffmpeg'):
        """
        Initialize HLS stream capture
        base_url: Base URL of the stream (e.g., https://example.com/stream/)
        playlist_url: Optional direct URL to .m3u8 playlist. If None, will try to find it
        ffmpeg_path: Path to ffmpeg executable
        """
        self.base_url = base_url.rstrip('/')
        self.playlist_url = playlist_url
        self.ffmpeg_path = ffmpeg_path
        self.frame_queue = Queue(maxsize=30)
        self.running = False
        self.current_segment = None
        
    def find_m3u8_playlist(self):
        """
        Try to locate the .m3u8 playlist by common patterns
        """
        common_patterns = [
            '/playlist.m3u8',
            '/index.m3u8',
            '/stream.m3u8',
            '/master.m3u8'
        ]
        
        print("Searching for HLS playlist...")
        for pattern in common_patterns:
            test_url = urljoin(self.base_url, pattern)
            try:
                response = requests.get(test_url, timeout=5)
                if response.status_code == 200 and response.text.strip().startswith('#EXTM3U'):
                    print(f"Found playlist at: {test_url}")
                    return test_url
            except requests.RequestException:
                continue
                
        return None

    def get_current_segment(self):
        """
        Get the latest segment from the HLS playlist
        """
        try:
            # If we don't have a playlist URL, try to find it
            if not self.playlist_url:
                self.playlist_url = self.find_m3u8_playlist()
                if not self.playlist_url:
                    raise Exception("Could not find HLS playlist")
            
            # Load and parse the playlist
            playlist = m3u8.load(self.playlist_url)
            
            # Get the latest segment
            if playlist.segments:
                latest_segment = playlist.segments[-1]
                segment_uri = latest_segment.uri
                
                # Handle relative URLs
                if not segment_uri.startswith('http'):
                    segment_uri = urljoin(self.base_url, segment_uri)
                
                return segment_uri
                
        except Exception as e:
            print(f"Error getting segment: {e}")
            return None

    def capture_stream(self):
        """
        Capture the HLS stream using ffmpeg
        """
        while self.running:
            try:
                segment_url = self.get_current_segment()
                if not segment_url:
                    print("No valid segment found, retrying...")
                    time.sleep(1)
                    continue
                
                command = [
                    self.ffmpeg_path,
                    '-i', segment_url,
                    '-f', 'image2pipe',
                    '-pix_fmt', 'bgr24',
                    '-vcodec', 'rawvideo',
                    '-hide_banner',
                    '-loglevel', 'error',
                    '-'
                ]
                
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                while self.running:
                    # Read raw video frame (assuming 352x240 resolution)
                    raw_frame = process.stdout.read(352 * 240 * 3)
                    if not raw_frame:
                        break
                    
                    frame = np.frombuffer(raw_frame, dtype=np.uint8)
                    try:
                        frame = frame.reshape((240, 352, 3))
                        if not self.frame_queue.full():
                            self.frame_queue.put(frame)
                    except ValueError as e:
                        print(f"Frame reshape error: {e}")
                        break
                
                process.terminate()
                    
            except Exception as e:
                print(f"Stream capture error: {e}")
                time.sleep(1)

    def display_frames(self):
        """
        Display captured frames
        """
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                
                # Create a resized display window (2x larger)
                display_frame = cv2.resize(frame, (704, 480))
                
                cv2.imshow('HLS Stream', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
            else:
                time.sleep(0.001)

    def run(self):
        """
        Start the stream capture and display
        """
        self.running = True
        
        capture_thread = Thread(target=self.capture_stream)
        display_thread = Thread(target=self.display_frames)
        
        capture_thread.start()
        display_thread.start()
        
        try:
            capture_thread.join()
            display_thread.join()
        except KeyboardInterrupt:
            self.running = False
        finally:
            cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # Example base URL - replace with your camera's base URL
    BASE_URL = "https://public.carsprogram.org/cameras/IN/INDOT_262__7ypTvHKbwMpXYJD"
    
    # Initialize and run the capture
    capture = HLSStreamCapture(BASE_URL)
    capture.run()