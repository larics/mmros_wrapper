#! /home/gideon/archiconda3/envs/mmdeploy/bin/python3.8

import numpy 
import rospy
import cv2

from PIL import ImageDraw, ImageOps, ImageFont
from PIL import Image as PILImage

from sensor_msgs.msg import Image, CompressedImage, Joy, PointCloud2

import random


def create_color_palette(type_="taco"):
    if type_ == "taco": 
        num_classes = 10
    if type_ == "coco": 
        num_classes = 90
        
    # Dictionary for color palette
    color_palette = {i: random_color() for i in range(1, num_classes + 1)}
    # Adding color for the background class (class id 0)
    color_palette[0] = (0, 0, 0)
    # Number of classes (excluding the background class)
    return color_palette

# Function to generate a random RGB color
def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

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

def plot_result(image_np, bboxes, labels, scores, score_threshold=0.2, plot_masks=False, plot_type="coco", color_dict = None, masks=None):
    """
    Plot bounding boxes on the given image.

    Parameters:
        image_np (np.ndarray): NumPy array representing the image (RGB format).
        bounding_boxes (list of lists): List of bounding boxes in COCO format [x, y, width, height].
        labels (list of str): List of corresponding labels for each bounding box.
        scores (list of float): List of confidence scores for each bounding box.
        score_threshold (float): The threshold below which bounding boxes will not be plotted.

    Returns:
        np.ndarray: NumPy array with plotted bounding boxes on the original image.
    """
    
    
    id_to_label = get_id_to_label(plot_type) 
        
    image_pil = PILImage.fromarray(image_np)
    
    if plot_masks:
        for box, mask, label, score in zip(bboxes, masks, labels, scores):
            if score >= score_threshold:
                draw = ImageDraw.Draw(image_pil)
                plot_bbox(box, label, draw, id_to_label, color_dict)
                color_ = label_to_color(label, color_dict)
                image_pil = overlay_binary_mask(image_np, image_pil, mask, color=color_, alpha_true=0.4)
    else:
        for box, label, score in zip(bboxes, labels, scores):
            if score >= score_threshold:
                plot_bbox(box, label, draw, id_to_label, color_dict)
    return image_pil
    
def overlay_binary_mask(img_np, pil_img, mask, color=(255, 0, 0), alpha_true=0.3):
    """
    Overlay a binary mask on the given image.

    Parameters:
        image_np (np.ndarray): NumPy array representing the image (RGB format).
        mask (np.ndarray): NumPy array representing the binary mask (2D array of True/False).
        color (tuple, optional): RGB color tuple to use for the True regions in the mask. Default is red (255, 0, 0).
        alpha_true (float, optional): Alpha value (transparency) for the True regions in the mask. Default is 0.3.

    Returns:
        np.ndarray: NumPy array with the binary mask overlaid on the original image.
    """
    assert img_np.shape[:2] == mask.shape, "Image and mask shape must match."

    mask_pil = PILImage.new("RGBA", pil_img.size)

    # Create a new mask with color values where mask is True
    mask_color = numpy.array(color + (int(255 * alpha_true),), dtype=numpy.uint8)
    mask_indices = numpy.where(mask)
    mask_pil_array = numpy.zeros((mask.shape[0], mask.shape[1], 4), dtype=numpy.uint8)
    mask_pil_array[mask_indices[0], mask_indices[1]] = mask_color

    mask_pil = PILImage.fromarray(mask_pil_array, "RGBA")

    return PILImage.alpha_composite(pil_img.convert("RGBA"), mask_pil)


def plot_bbox(box, label, draw, id_to_label, color_dict, score=None):
    x, y, width, height = box
    draw.rectangle([x, y, width, height], outline=label_to_color(label, color_dict), width=3)
    draw.text((x, y), f"{id_to_label[label]}: {score:.2f}" if score is not None else id_to_label[label], fill=label_to_color(label, color_dict))
    return draw

def convert_to_rects(bboxes, scores, score_threshold=0.5): 
    rects = []
    for score, bbox in zip(scores, bboxes):
        if score > score_threshold:
            x, y, width, height = bbox
            rects.append([x, y, x+width, y+height])
    return rects
    
def filter_bboxes(bboxes, scores, score_threshold=0.5): 
    rects = []
    for score, bbox in zip(scores, bboxes):
        if score > score_threshold:
            rects.append(bbox)
    return rects
    
def convert_tensor_to_array(result_tensor): 
    return result_tensor.pred_sem_seg.data.detach().cpu().numpy()   


def label_to_color(label, color_dict=None):
    
    if color_dict:
        return color_dict[label]
    else: 
        
        color_dict = {
            0: (255, 204, 204),    # Light Pastel Red
            1: (204, 255, 204),    # Light Pastel Green
            2: (204, 204, 255),    # Light Pastel Blue
            3: (255, 255, 204),    # Light Pastel Yellow
            4: (255, 224, 179),    # Light Pastel Orange
            5: (221, 160, 221),    # Light Pastel Purple
            6: (204, 255, 255),    # Light Pastel Cyan
            7: (255, 204, 255),    # Light Pastel Magenta
            8: (204, 255, 204),    # Light Pastel Green (same as 1)
            9: (244, 164, 96),     # Light Pastel Orange (a different shade of orange)
            10: (240, 230, 140),   # Light Pastel Olive
            }
        # Calculate the color index based on the label ID
        return color_dict[label]


def get_id_to_label(type_):
    
    if type_ == "coco": 
        # Dictionary for mapping ids to COCO labels
        id_to_label = {0: "background",
            1: "person",
            2: "bicycle",
            3: "car",
            4: "motorcycle",
            5: "airplane",
            6: "bus",
            7: "train",
            8: "truck",
            9: "boat",
            10: "traffic light",
            11: "fire hydrant",
            12: "stop sign",
            13: "parking meter",
            14: "bench",
            15: "bird",
            16: "cat",
            17: "dog",
            18: "horse",
            19: "sheep",
            20: "cow",
            21: "elephant",
            22: "bear",
            23: "zebra",
            24: "giraffe",
            25: "backpack",
            26: "umbrella",
            27: "handbag",
            28: "tie",
            29: "suitcase",
            30: "frisbee",
            31: "skis",
            32: "snowboard",
            33: "sports ball",
            34: "kite",
            35: "baseball bat",
            36: "baseball glove",
            37: "skateboard",
            38: "surfboard",
            39: "tennis racket",
            40: "bottle",
            41: "wine glass",
            42: "cup",
            43: "fork",
            44: "knife",
            45: "spoon",
            46: "bowl",
            47: "banana",
            48: "apple",
            49: "sandwich",
            50: "orange",
            51: "broccoli",
            52: "carrot",
            53: "hot dog",
            54: "pizza",
            55: "donut",
            56: "cake",
            57: "chair",
            58: "couch",
            59: "potted plant",
            60: "bed",
            61: "dining table",
            62: "toilet",
            63: "tv",
            64: "laptop",
            65: "mouse",
            66: "remote",
            67: "keyboard",
            68: "cell phone",
            69: "microwave",
            70: "oven",
            71: "toaster",
            72: "sink",
            73: "refrigerator",
            74: "book",
            75: "clock",
            76: "vase",
            77: "scissors",
            78: "teddy bear",
            79: "hair drier",
            80: "toothbrush",
            81: "hair brush"}

    if type_ == "taco":  
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
                       
    return id_to_label
