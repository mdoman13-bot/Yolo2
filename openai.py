#!/usr/bin/env python3
"""
4‑stream HLS viewer (side‑by‑side or 2×2 grid)

Requirements:
    pip install ffmpeg-python opencv-python
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple
import os

import cv2
import numpy as np
import ffmpeg  # pip install ffmpeg-python

# --------------------------------------------------------------------------- #
# Configuration -------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

STREAM_URLS = [
    "https://skysfs3.trafficwise.org/rtplive/INDOT_257_IlQ0iAJPF3zCjVhF/playlist.m3u8",
    "https://skysfs4.trafficwise.org/rtplive/INDOT_262__7ypTvHKbwMpXYJD/playlist.m3u8",
    "https://skysfs3.trafficwise.org/rtplive/INDOT_703_I3RqqDqcbqI1A_Z3/playlist.m3u8",
    "https://skysfs3.trafficwise.org/rtplive/INDOT_261_B6pE8gVw3RJ7YdXn/playlist.m3u8",
]

# Desired resolution for each stream (width, height)
TARGET_SIZE = (640, 360)          # you can change it to whatever fits your monitor

# How we lay out the streams
LAYOUT = "grid"  # options: "hstack", "vstack", "grid"

WINDOW_NAME = "4‑stream viewer"
WINDOW_WIDTH, WINDOW_HEIGHT = TARGET_SIZE[0] * 2, TARGET_SIZE[1] * 2

ffmpeg_path = r"C:\Program Files\FFMPEG\ffmpeg-7.0.2-essentials_build\bin\ffmpeg.exe"
# --------------------------------------------------------------------------- #
# Helper functions ------------------------------------------------------------ #
# --------------------------------------------------------------------------- #

def spawn_ffmpeg_reader(url: str, width: int, height: int) -> subprocess.Popen:
    """
    Starts an FFmpeg process that reads the given HLS URL and outputs raw RGB24 frames.
    Returns the Popen object (you can read from p.stdout).
    """
    # Build ffmpeg command
    cmd = (
        ffmpeg
        .input(url, timeout=5)
        .output(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s=f"{width}x{height}",
            r=30,          # try to read at 30 fps; FFmpeg will drop frames if it can't keep up
        )
        .global_args("-hide_banner", "-loglevel", "error")  # suppress noise
    )
    process = cmd.run_async(
        pipe_stdout=True,
        pipe_stderr=subprocess.PIPE,
        cmd=ffmpeg_path  # use the specified ffmpeg binary
    )
    return process

def read_frame(proc: subprocess.Popen, width: int, height: int) -> np.ndarray | None:
    """
    Reads a single frame from the given FFmpeg stdout.
    Returns a NumPy array of shape (height, width, 3) in RGB format,
    or None if we hit EOF / error.
    """
    frame_size = width * height * 3
    raw_bytes = proc.stdout.read(frame_size)
    if len(raw_bytes) != frame_size:
        return None
    # Convert to numpy array and reshape
    img = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((height, width, 3))
    return img

def compose_frames(frames: List[np.ndarray], layout: str="grid") -> np.ndarray:
    """
    Combines four frames into a single image.
    """
    if len(frames) != 4:
        raise ValueError("Expected exactly 4 frames")

    # Pad missing frames with black
    for i in range(4):
        if frames[i] is None:
            h, w = TARGET_SIZE[1], TARGET_SIZE[0]
            frames[i] = np.zeros((h, w, 3), dtype=np.uint8)

    if layout == "hstack":
        top = cv2.hconcat(frames[:2])
        bottom = cv2.hconcat(frames[2:])
        return cv2.vconcat([top, bottom])

    elif layout == "vstack":
        left = cv2.vconcat(frames[:2])
        right = cv2.vconcat(frames[2:])
        return cv2.hconcat([left, right])

    else:  # grid
        top = cv2.hconcat(frames[:2])
        bottom = cv2.hconcat(frames[2:])
        return cv2.vconcat([top, bottom])


# --------------------------------------------------------------------------- #
# Main --------------------------------------------------------------           #
# --------------------------------------------------------------------------- #

def main():
    width, height = TARGET_SIZE

    # Start one FFmpeg process per stream
    procs = [spawn_ffmpeg_reader(url, width, height) for url in STREAM_URLS]

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

    print("Press 'q' to quit.")
    try:
        while True:
            frames = []
            # Read one frame from each stream
            for proc in procs:
                frame = read_frame(proc, width, height)
                frames.append(frame)

            # Compose into a single image
            combined = compose_frames(frames, layout=LAYOUT)

            # Show
            cv2.imshow(WINDOW_NAME, combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Clean up: kill all FFmpeg processes
        for proc in procs:
            proc.kill()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
