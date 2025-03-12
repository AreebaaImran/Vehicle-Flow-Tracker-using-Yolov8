from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

image_path = ("D:\Data Science With AI Course\Final Assignment\Final Assignment\sample_image.png")
output_path = ("D:\Data Science With AI Course\Final Assignment\Final Assignment")

results = model.predict(image_path, save=True, save_dir=output_path)