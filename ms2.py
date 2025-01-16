import cv2
import torch
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import threading
import math

# Load the YOLO model
model = YOLO('models/yolo11x')  # Replace with the path to your model
model.to('cuda') if torch.cuda.is_available() else model.to('cpu') # Use GPU for inference
# To get this to work, I had to pip uninstall opencv-python then pip install opencv-python
# Open the live stream
# keystone -> 31
# stream_url = "https://skysfs3.trafficwise.org/rtplive/INDOT_257_IlQ0iAJPF3zCjVhF/playlist.m3u8"

# stream_url = "https://skysfs4.trafficwise.org/rtplive/INDOT_262__7ypTvHKbwMpXYJD/playlist.m3u8"
# Westfield crossover with state road 32
# stream_url = "https://skysfs3.trafficwise.org/rtplive/INDOT_703_I3RqqDqcbqI1A_Z3/playlist.m3u8"

# stream_url = "https://skysfs3.trafficwise.org/rtplive/INDOT_261_B6pE8gVw3RJ7YdXn/playlist.m3u8"
# 31 and 151st
# stream_url = "https://skysfs4.trafficwise.org/rtplive/INDOT_260_l6kGZjfYqqL9vgE9/playlist.m3u8"


# stream_url = "https://skysfs3.trafficwise.org/rtplive/INDOT_257_IlQ0iAJPF3zCjVhF/playlist.m3u8"
# keystone to allisonville
stream_url = "https://skysfs4.trafficwise.org/rtplive/INDOT_32_8Aa6tXj3dYK35_P2/playlist.m3u8"
# stream_url = "https://skysfs3.trafficwise.org/rtplive/INDOT_579_6c5KpTSKIJWts7HU/playlist.m3u8"
# stream_url = "https://skysfs3.trafficwise.org/rtplive/INDOT_591_ikD7yYJFi8SlJxfy/playlist.m3u8"

# Replace single stream with multiple streams
stream_urls = [
    "https://skysfs3.trafficwise.org/rtplive/INDOT_257_IlQ0iAJPF3zCjVhF/playlist.m3u8",
    "https://skysfs4.trafficwise.org/rtplive/INDOT_260_l6kGZjfYqqL9vgE9/playlist.m3u8"
]

frames_dict = {}
should_stop = False

def process_stream(url):
    global should_stop
    cap = cv2.VideoCapture(url)
    while not should_stop:
        ret, frame = cap.read()
        if not ret:
            continue
        results = model(frame)
        # Example heatmap placeholder, replace as needed
        heatmap = np.zeros_like(frame)
        # Store combined data
        frames_dict[url] = (results[0].plot() if results else frame, heatmap)
    cap.release()

def find_grid_dims(n):
    if n <= 4:
        return (1, n)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    return rows, cols

def tile_frames(items):
    if not items:
        return None
    rows, cols = find_grid_dims(len(items))
    combined_pairs = []
    for (f, h) in items:
        f_resized = cv2.resize(f, (320, 240))
        h_resized = cv2.resize(h, (320, 240))
        combined_pairs.append(np.hstack((f_resized, h_resized)))
    blank = np.zeros_like(combined_pairs[0])
    while len(combined_pairs) < rows * cols:
        combined_pairs.append(blank)
    final_rows = []
    idx = 0
    for r in range(rows):
        row = combined_pairs[idx]
        for c in range(1, cols):
            row = np.hstack((row, combined_pairs[idx + c]))
        final_rows.append(row)
        idx += cols
    return np.vstack(final_rows)

threads = []
for url in stream_urls:
    t = threading.Thread(target=process_stream, args=(url,))
    t.start()
    threads.append(t)

while True:
    tiled = tile_frames(list(frames_dict.values()))
    if tiled is not None:
        cv2.imshow("Combined Streams", tiled)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        should_stop = True
        break

for t in threads:
    t.join()

cv2.destroyAllWindows()