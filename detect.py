from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

video_path= r"D:\Data Science With AI Course\Final Assignment\Final Assignment\sample_video.mp4"
cap=cv2.VideoCapture(video_path)

# Store detection results for each frame
results_list = []

# Define output video file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
out = cv2.VideoWriter("detect.mp4", fourcc, 25.0, (3840, 2160))  # (filename, codec, FPS, resolution)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    # Perform object detection on the frame
    results = model(frame)
    # results_list.append(results)  # Store detection results
    
    # Draw bounding boxes on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            conf = box.conf[0].item()

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # cv2.imshow("All Detections", frame)
    out.write(frame)  # Save the frame to output video

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

out.release()
cap.release()
cv2.destroyAllWindows()