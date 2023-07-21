#! /home/gideon/archiconda3/envs/mmdeploy/bin/python3.8

import numpy 
import rospy
import cv2

from PIL import ImageDraw, ImageOps, ImageFont
from PIL import Image as PILImage

from sensor_msgs.msg import Image, CompressedImage, Joy, PointCloud2


def convert_pil_to_ros_img(img, header):
     img = img.convert('RGB')
     msg = Image()
     msg.header = header
     msg.height = img.height
     msg.width = img.width
     msg.encoding = "rgb8"
     msg.is_bigendian = False
     msg.step = 3 * img.width
     msg.data = numpy.array(img).tobytes()
     return msg


def plot_bboxes(image_np, bounding_boxes, labels):
    """
    Plot bounding boxes on the given image.

    Parameters:
        image_np (np.ndarray): NumPy array representing the image (RGB format).
        bounding_boxes (list of lists): List of bounding boxes in COCO format [x, y, width, height].
        labels (list of str): List of corresponding labels for each bounding box.

    Returns:
        np.ndarray: NumPy array with plotted bounding boxes on the original image.
    """
    id_to_label = {0: 'Other',
               1: 'Plastic bag',
               2: 'Bottle Cap',
               3: 'Bottle',
               4: 'Cup', 
               5: 'Lid',
               6: 'Can', 
               7: 'Pop tab', 
               8: 'Straw',
               9: 'Cigarette'}

    image_pil = PILImage.fromarray(image_np)
    draw = ImageDraw.Draw(image_pil)

    for box, label in zip(bounding_boxes, labels):
        x, y, width, height = box
        draw.rectangle([x, y, x + width, y + height], outline="red", width=3)
        draw.text((x, y), id_to_label[label], fill="red")

    return image_pil


