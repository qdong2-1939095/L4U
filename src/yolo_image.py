# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
from resize import resize_image
from PIL import Image

CONFIDENCE = 0.5
THRESHOLD = 0.3
LABELS = []
COLORS = None

def parse_arguments():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image",
        help="path to input image")
    ap.add_argument("-y", "--yolo", default='yolo',
        help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
        help="threshold when applying non-maxima suppression")
    return vars(ap.parse_args())

def load_yolo(yolo_dir):
    global LABELS, COLORS
    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([yolo_dir, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([yolo_dir, "yolov3.weights"])
    configPath = os.path.sep.join([yolo_dir, "yolov3.cfg"])
    # load our YOLO object detector trained on COCO dataset (80 classes)
    # print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    return net

def label_image(image, net):
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    # show timing information on YOLO
    # print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    return layerOutputs

def draw_box(image, layerOutputs, ii):
    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    (H, W) = image.shape[:2]    # spatial dimensions
    # print("image: ", H, W)

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONFIDENCE:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # print((centerX, centerY, width, height))
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)
            # print(LABELS[classIDs[i]], confidences[i])
            if LABELS[classIDs[i]] == 'person':
                # cv2.imwrite(f"output_ski/labeled/boxed{ii}.jpg", image)
                # # cv2.imshow("Image", image)
                # cv2.waitKey(0)
                return (x, y, w, h, W, H)
    # # show the output image
    # cv2.imwrite(f"output_ski/boxed{ii}.jpg", image)
    # # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    print("    no person detected!")
    return None, None, None, None, None, None
    

def main():
    args = parse_arguments()
    global CONFIDENCE, THRESHOLD
    CONFIDENCE = args["confidence"]
    THRESHOLD = args["threshold"]
    for i in range(231):
        image_path = f"input/frame{i}.jpg"
        print("processing", image_path, "...")
        image = cv2.imread(image_path)   # input image
        net = load_yolo(args["yolo"])
        layerOutput = label_image(image, net)
        left_x, top_y, w, h, im_width, im_height = draw_box(image, layerOutput, i)
        if left_x == None:
            continue
        left = left_x
        right = left_x + w
        top = top_y
        bottom = top_y + h
        res = resize_image(left, right, top, bottom, im_width, im_height)
        img = Image.open(image_path)
        cropped = img.crop((res[0], res[2], res[1], res[3]))
        cropped.save(f"temp/frame{i}.jpg")
        
        im = Image.open(f"temp/frame{i}.jpg")
        im.resize((im_width, im_height)).save(f"output_ski/yolo_output/frame{i}.jpg")

if __name__ == "__main__":
    main()