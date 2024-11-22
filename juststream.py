import cv2
import numpy as np

# List of stream URLs
stream_urls = [
    "https://skysfs3.trafficwise.org/rtplive/INDOT_257_IlQ0iAJPF3zCjVhF/playlist.m3u8",
    "https://skysfs4.trafficwise.org/rtplive/INDOT_262__7ypTvHKbwMpXYJD/playlist.m3u8",
    "https://skysfs3.trafficwise.org/rtplive/INDOT_703_I3RqqDqcbqI1A_Z3/playlist.m3u8",
    "https://skysfs3.trafficwise.org/rtplive/INDOT_261_B6pE8gVw3RJ7YdXn/playlist.m3u8"
]

# Number of streams
num_streams = len(stream_urls)

# Open video streams
caps = [cv2.VideoCapture(url) for url in stream_urls]

# Function to get the frame from a video capture object
def get_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return np.zeros((240, 320, 3), np.uint8)
    return frame

# Get the resolution of each stream
resolutions = []
for cap in caps:
    ret, frame = cap.read()
    if ret:
        resolutions.append((frame.shape[1], frame.shape[0]))  # (width, height)
    else:
        resolutions.append((320, 240))  # Default resolution if stream is not available

# Determine the lowest resolution
min_width = min(res[0] for res in resolutions)
min_height = min(res[1] for res in resolutions)

while True:
    frames = [get_frame(cap) for cap in caps]

    # Resize frames to the lowest resolution
    resized_frames = [cv2.resize(frame, (min_width, min_height)) for frame in frames]

    # Create a grid of frames
    if num_streams == 4:
        top_row = np.hstack(resized_frames[:2])
        bottom_row = np.hstack(resized_frames[2:])
        grid = np.vstack((top_row, bottom_row))
    else:
        grid = np.hstack(resized_frames)

    # Display the grid
    cv2.imshow('Stream Grid', grid)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video captures and close windows
for cap in caps:
    cap.release()
cv2.destroyAllWindows()