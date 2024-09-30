import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("media/cars.m4v")

model = YOLO("yolo11n.pt")
freq_dict = {}

# Get the dimensions of the input video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a heatmap to show the average location of the most detected objects
heatmap = np.zeros((height, width))

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        results = model(frame, device="mps")
        result = results[0]
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        classes = np.array(result.boxes.cls.cpu(), dtype="int")
        scores = np.array(result.boxes.conf.cpu(), dtype="float")

        for cls, score in zip(classes, scores):
            label = model.names[cls]
            if label not in freq_dict:
                freq_dict[label] = {"count": 1, "score_sum": score}
            else:
                freq_dict[label]["count"] += 1
                freq_dict[label]["score_sum"] += score

        for cls, bbox in zip(classes, bboxes):
            (x, y, x2, y2) = bbox
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
            cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)

            # Increment heatmap values for each detected object
            heatmap[y:y2, x:x2] += 1

        cv2.imshow("Img", frame)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break
except KeyboardInterrupt:
    print("Program interrupted by user.")
finally:
    # Normalize heatmap values between 0 and 1
    heatmap = heatmap / heatmap.max()

    # Convert the dictionary to two lists for plotting
    tokens = list(freq_dict.keys())
    freqs = [freq_dict[token]["count"] for token in tokens]
    scores = [freq_dict[token]["score_sum"] / freq_dict[token]["count"] for token in tokens]

    # Save frequency dictionary to file
    with open("freq_dict.txt", "a") as f:
        f.write(str(freq_dict) + "\n")
    f.close()

    # Create frequency and confidence score bar charts
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(tokens, freqs)
    ax.set_xlabel('Predicted Tokens')
    ax.set_ylabel('Frequency')

    # Create a second axis for the confidence scores
    ax2 = ax.twinx()
    ax2.set_ylim(0, 1)  # Set y-axis limits to match confidence score range

    # Create a bar chart for the confidence scores, with a different color
    ax2.bar(tokens, scores, alpha=0.5, color='r', label='Confidence Score')
    ax2.set_ylabel('Average Confidence Score', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Display the heatmap
    plt.figure()
    plt.imshow(heatmap, alpha=0.5, cmap='jet', interpolation='bilinear')
    plt.axis('off')
    plt.title('Object Detection Heatmap')
    plt.show()

cap.release()
cv2.destroyAllWindows()
