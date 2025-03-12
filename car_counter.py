from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

video_path = r"D:\Data Science With AI Course\Final Assignment\Final Assignment\sample_video.mp4"
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("car_count.mp4", fourcc, fps, (width, height))

cars_in = 0
cars_out = 0
roi_y = int(height * 0.8)
tracked_cars = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection + tracking
    results = model.track(frame, persist=True)

    cv2.line(frame, (0, roi_y), (width, roi_y), (0, 255, 255), 7)

    for result in results:
        for box in result.boxes:
            track_id = int(box.id[0]) if box.id is not None else None  # Unique ID
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            conf = box.conf[0].item()

            if label == "car" and track_id is not None:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, f"ID {track_id} {label} {conf:.2f}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                center_y = (y1 + y2) // 2

                # Check if this car was previously detected
                if track_id in tracked_cars:
                    prev_y = tracked_cars[track_id]

                    if prev_y < roi_y and center_y >= roi_y:
                        cars_in += 1  # Moving DOWN (IN)
                    elif prev_y > roi_y and center_y <= roi_y:
                        cars_out += 1  # Moving UP (OUT)

                tracked_cars[track_id] = center_y  # Update position
  
    # Display car count
    cv2.putText(frame, f"Coming In: {cars_in}", (width // 2 - 200, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
    cv2.putText(frame, f"Going Out: {cars_out}", (width // 2 - 200, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()