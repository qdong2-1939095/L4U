import cv2
import numpy as np
import os

from os.path import isfile, join

def convert_frames_to_video(in_dir, out_file, fps):
    frame_array = []
    files = os.listdir(in_dir)

    for i in range(231):
        filename = in_dir + f"frame{i}.jpg"
        #reading each files
        if not os.path.isfile(filename):
            continue
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*"DIVX"), fps, size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()

if __name__=="__main__":
    convert_frames_to_video("output/", "centered.mp4", 30)
