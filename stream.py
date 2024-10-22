import cv2
import ffmpeg
# Replace with the actual URL of the video stream
video_url = "https://skysfs4.trafficwise.org/rtplive/INDOT_262__7ypTvHKbwMpXYJD/media_w583159886_46.ts"

# Open the video stream
cap = cv2.VideoCapture(video_url)

# stream = ffmpeg.input(video_url)
if not cap.isOpened():
    print("Error: Cannot open video stream")
    exit()

# Stream video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Display the frame
    cv2.imshow('Video Stream', frame)
    
    # Exit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
