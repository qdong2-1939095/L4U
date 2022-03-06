import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from PIL import ImageDraw

import time

def display_image(image):
    fig = plt.figure(figsize=(20, 15))
    plt.grid(False)
    plt.imshow(image)
    plt.show()

def run_detector(detector, path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)

    converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    start_time = time.time()
    result = detector(converted_img)
    result = {key:value.numpy() for key,value in result.items()}
    # print(result)
    print("time:", time.time() - start_time)

    image = Image.fromarray(np.uint8(img.numpy())).convert("RGB")
    indices = [i for i, x in enumerate(result["detection_class_labels"]) if x == 69]  # type "person"
    boxes = np.take(result["detection_boxes"], indices, axis=0)
    scores = np.take(result["detection_scores"], indices)
    i = np.argmax(scores)
    ymin, xmin, ymax, xmax = tuple(boxes[i])
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)

    draw = ImageDraw.Draw(image)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], \
            width=4, fill=128)
    display_image(image)


if __name__ == "__main__":
    image_path = "input/frame0.jpg"

    module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
    detector = hub.load(module_handle).signatures["default"]
    run_detector(detector, image_path)
