import cv2

video_path = r"D:\Data Science With AI Course\Final Assignment\Final Assignment\sample_video.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)  # Get FPS of input video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get width
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get height
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames

print(f"FPS: {fps}")
print(f"Resolution: {width}x{height}")
print(f"Total Frames: {frame_count}")

cap.release()
