import os
import tensorflow_hub as hub
from PIL import Image

from detector import run_detector
from resize import resize_image

if not os.path.exists("temp"):
    os.makedirs("temp")
if not os.path.exists("output"):
    os.makedirs("output")

module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
detector = hub.load(module_handle).signatures["default"]

images = os.listdir("input")
for i, x in enumerate(images):
    image_path = f"input/{x}"
    left, right, top, bottom, im_width, im_height = run_detector(detector, image_path)

    res = resize_image(left, right, top, bottom, im_width, im_height)
    img = Image.open(image_path)
    cropped = img.crop((res[0], res[2], res[1], res[3]))
    cropped.save(f"temp/{x}")
    
    im = Image.open(f"temp/{x}")
    im.resize((im_width, im_height)).save(f"output/{x}")
