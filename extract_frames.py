import cv2
import os

if not os.path.exists("input"):
    os.makedirs("input")

vidcap = cv2.VideoCapture("test_video.mp4")
success, image = vidcap.read()
count = 0
while success:
    cv2.imwrite(f"input/frame{count}.jpg", image)
    success, image = vidcap.read()
    count += 1
