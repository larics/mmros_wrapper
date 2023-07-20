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
