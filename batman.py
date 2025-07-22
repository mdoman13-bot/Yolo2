import tkinter as tk
import cv2
import numpy as np
import math
import time

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

stream_urls = [
    'https://skysfs3.trafficwise.org/rtplive/INDOT_257_IlQ0iAJPF3zCjVhF/playlist.m3u8',
    'https://skysfs4.trafficwise.org/rtplive/INDOT_262__7ypTvHKbwMpXYJD/playlist.m3u8',
    'https://skysfs3.trafficwise.org/rtplive/INDOT_703_I3RqqDqcbqI1A_Z3/playlist.m3u8',
    'https://skysfs3.trafficwise.org/rtplive/INDOT_261_B6pE8gVw3RJ7YdXn/playlist.m3u8',
    'https://skysfs4.trafficwise.org/rtplive/INDOT_260_l6kGZjfYqqL9vgE9/playlist.m3u8',
    'https://skysfs4.trafficwise.org/rtplive/INDOT_32_8Aa6tXj3dYK35_P2/playlist.m3u8',
    'https://skysfs3.trafficwise.org/rtplive/INDOT_579_6c5KpTSKIJWts7HU/playlist.m3u8',
    'https://skysfs3.trafficwise.org/rtplive/INDOT_591_ikD7yYJFi8SlJxfy/playlist.m3u8'
]

frames_dict = {}
status_dict = {}
caps = []

def open_all_streams():
    for idx, url in enumerate(stream_urls):
        cap = cv2.VideoCapture(url)
        caps.append(cap)
        status_dict[idx] = "Connecting"

def read_all_streams():
    for idx, cap in enumerate(caps):
        if not cap.isOpened():
            status_dict[idx] = "Offline"
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(frame, 'No Signal', (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            frames_dict[idx] = frame
            # Try to reconnect
            cap.release()
            caps[idx] = cv2.VideoCapture(stream_urls[idx])
            status_dict[idx] = "Reconnecting"
            continue
        ret, frame = cap.read()
        if not ret or frame is None:
            status_dict[idx] = "Offline"
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(frame, 'No Signal', (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            frames_dict[idx] = frame
            # Try to reconnect
            cap.release()
            caps[idx] = cv2.VideoCapture(stream_urls[idx])
            status_dict[idx] = "Reconnecting"
        else:
            frame = cv2.resize(frame, (320, 240))
            frames_dict[idx] = frame
            status_dict[idx] = "Online"

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
    labels = []
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
        labels.append(label)
    return labels

def update_status_labels(labels):
    for idx, label in enumerate(labels):
        if idx < len(stream_urls):
            status = status_dict.get(idx, "Idle")
            base_text = streams[idx]
            label.config(text=f"{base_text}\n[{status}]")
            if status == "Online":
                label.config(fg="green")
            elif status in ("Connecting", "Reconnecting"):
                label.config(fg="orange")
            elif status == "Offline":
                label.config(fg="red")
            else:
                label.config(fg="gray")
        else:
            label.config(text=streams[idx], fg="gray")

def update_opencv_window():
    frames = [frames_dict.get(i, np.zeros((240, 320, 3), dtype=np.uint8)) for i in range(len(stream_urls))]
    tiled = tile_frames(frames)
    if tiled is not None:
        cv2.imshow('Security Camera Grid', tiled)
    # Use a very short waitKey to keep window responsive but not block Tk
    if cv2.waitKey(1) & 0xFF == ord('q'):
        root.quit()
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()
        return
    root.after(50, update_opencv_window)  # Update OpenCV window every 50ms

def main_loop():
    read_all_streams()
    update_status_labels(labels)
    root.after(200, main_loop)  # ~5 FPS

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Batcave Security Grid")
    labels = create_grid(root, streams, columns=3)
    open_all_streams()
    root.after(100, main_loop)
    root.after(200, update_opencv_window)
    root.mainloop()
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()