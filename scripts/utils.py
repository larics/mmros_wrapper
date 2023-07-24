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


def plot_result(image_np, bboxes, labels, scores, score_threshold=0.2, plot_masks=False, masks=None):
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
    
    if plot_masks:
        for box, mask, label, score in zip(bboxes, masks, labels, scores):
            if score >= score_threshold:
                plot_bbox(box, label, draw, id_to_label)
                color_ = label_to_color(label)
                image_pil = overlay_binary_mask(image_np, image_pil, mask, color=color_, alpha_true=0.4)
    else:
        for box, label, score in zip(bboxes, labels, scores):
            if score >= score_threshold:
                plot_bbox(box, label, draw, id_to_label)
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
    

def plot_bbox(box, label, draw, id_to_label, score=None):
    x, y, width, height = box
    draw.rectangle([x, y, width, height], outline=label_to_color(label), width=3)
    draw.text((x, y), f"{id_to_label[label]}: {score:.2f}" if score is not None else id_to_label[label], fill=label_to_color(label))
 


def convert_tensor_to_array(result_tensor): 
    return result_tensor.pred_sem_seg.data.detach().cpu().numpy()   


def label_to_color(label):
    color_palette = {
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
    color_index = label % len(color_palette)

    # Return the RGB color tuple for the corresponding index
    return color_palette[color_index]



