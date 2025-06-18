import tkinter as tk
import cv2
import numpy as np
import math
import threading

# List of streams (commented out ones included as strings)
streams = [
    "Camera 1 - Active",
    "Camera 2 - Active",
    "# Camera 3 - Offline",
    "Camera 4 - Active",
    "# Camera 5 - Maintenance",
    "Camera 6 - Active",
    "Camera 7 - Active",
    "# Camera 8 - Disconnected",
    "Camera 9 - Active"
]

# All stream URLs, including commented out ones from multistream.py
stream_urls = [
    'https://skysfs3.trafficwise.org/rtplive/INDOT_257_IlQ0iAJPF3zCjVhF/playlist.m3u8',  # keystone -> 31
    'https://skysfs4.trafficwise.org/rtplive/INDOT_262__7ypTvHKbwMpXYJD/playlist.m3u8',  # Westfield crossover with state road 32
    'https://skysfs3.trafficwise.org/rtplive/INDOT_703_I3RqqDqcbqI1A_Z3/playlist.m3u8',
    'https://skysfs3.trafficwise.org/rtplive/INDOT_261_B6pE8gVw3RJ7YdXn/playlist.m3u8',
    'https://skysfs4.trafficwise.org/rtplive/INDOT_260_l6kGZjfYqqL9vgE9/playlist.m3u8',  # 31 and 151st
    'https://skysfs4.trafficwise.org/rtplive/INDOT_32_8Aa6tXj3dYK35_P2/playlist.m3u8',   # keystone to allisonville
    'https://skysfs3.trafficwise.org/rtplive/INDOT_579_6c5KpTSKIJWts7HU/playlist.m3u8',
    'https://skysfs3.trafficwise.org/rtplive/INDOT_591_ikD7yYJFi8SlJxfy/playlist.m3u8'
]

frames_dict = {}
should_stop = False

def process_stream(url):
    global should_stop
    cap = cv2.VideoCapture(url)
    while not should_stop:
        ret, frame = cap.read()
        if not ret:
            # If failed, show a black frame with text
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(frame, 'No Signal', (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
            frame = cv2.resize(frame, (320, 240))
        frames_dict[url] = frame
    cap.release()

def find_grid_dims(n):
    n = min(n, 16)
    rows = int(math.ceil(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    return rows, cols

def tile_frames(frames):
    if not frames:
        return None
    rows, cols = find_grid_dims(len(frames))
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    while len(frames) < rows * cols:
        frames.append(blank)
    grid = []
    idx = 0
    for r in range(rows):
        row = frames[idx]
        for c in range(1, cols):
            row = np.hstack((row, frames[idx + c]))
        grid.append(row)
        idx += cols
    return np.vstack(grid)

def create_grid(root, items, columns=3):
    for idx, stream in enumerate(items):
        row = idx // columns
        col = idx % columns
        label_text = stream
        if stream.strip().startswith("#"):
            fg_color = "gray"
        else:
            fg_color = "black"
        label = tk.Label(root, text=label_text, width=25, height=5, borderwidth=2, relief="groove", fg=fg_color)
        label.grid(row=row, column=col, padx=5, pady=5)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Batcave Security Grid")
    create_grid(root, streams, columns=3)
    root.mainloop()

threads = []
for url in stream_urls:
    t = threading.Thread(target=process_stream, args=(url,))
    t.start()
    threads.append(t)

try:
    while True:
        frames = list(frames_dict.values())
        tiled = tile_frames(frames)
        if tiled is not None:
            cv2.imshow('Security Camera Grid', tiled)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            should_stop = True
            break
finally:
    for t in threads:
        t.join()
    cv2.destroyAllWindows()