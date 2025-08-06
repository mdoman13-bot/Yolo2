# filepath: c:\Users\mod\Downloads\github\yolo2\openai.py
import subprocess
try:
    subprocess.run(
        [rffmpeg_path = r"C:\Program Files\FFMPEG\ffmpeg-7.0.2-essentials_build\bin\ffmpeg.exe", "-version"],
        check=True
    )
except Exception as e:
    print("Subprocess test failed:", e)
# ...existing code...