#!/usr/bin/env python3

import numpy 
import rospy
import cv2

from PIL import ImageDraw, ImageOps, ImageFont
from PIL import Image as PILImage

from sensor_msgs.msg import Image, CompressedImage, Joy, PointCloud2

import random

def create_color_palette(type_="taco"):
    if type_ == "taco": 
        num_classes = 2
    if type_ == "coco": 
        num_classes = 90
        
    # Dictionary for color palette
    color_palette = {i: random_color() for i in range(1, num_classes + 1)}
    # Adding color for the background class (class id 0)
    color_palette[0] = (255, 204, 204)
    color_palette[1] = (204, 255, 204)
    color_palette[2] = (204, 204, 255)
    # Number of classes (excluding the background class)
    return color_palette

# Function to generate a random RGB color
def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# TODO: Put these into one conversions script! 
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

def convert_np_array_to_ros_img_msg(numpy_array, encoding="mono8"):
    ros_image_msg = Image()
    ros_image_msg.height = numpy_array.shape[0]
    ros_image_msg.width = numpy_array.shape[1]
    ros_image_msg.encoding = encoding  # Adjust the encoding based on your array's format
    ros_image_msg.is_bigendian = 0
    ros_image_msg.step = ros_image_msg.width * 3  # Assuming 3 channels (adjust accordingly)
    ros_image_msg.data = numpy_array.tobytes()
    return ros_image_msg

def ros_image_to_numpy(image_msg):
    # Extract image data from the ROS Image message
    image_data = image_msg.data
    # Get image dimensions
    width = image_msg.width
    height = image_msg.height
    # Get the image encoding
    encoding = image_msg.encoding
    # Create a NumPy array from the image data
    numpy_array = numpy.frombuffer(image_data, dtype=numpy.uint8)
    # Reshape the array to match the image dimensions
    numpy_array = numpy_array.reshape((height, width, -1))
    return numpy_array

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
    
    
    start_time = rospy.Time.now()
    id_to_label = get_id_to_label(plot_type) 
        
    image_pil = PILImage.fromarray(image_np)
    draw = ImageDraw.Draw(image_pil)
    if plot_masks:
        assert len(bboxes) == len(masks) == len(labels) == len(scores), "Mismatch in list lengths"
        for box, mask, label, score in zip(bboxes, masks, labels, scores):
            if score >= score_threshold:
                color_ = label_to_color(label, color_dict)
                image_pil = overlay_binary_mask(image_np, image_pil, mask, color=color_, alpha_true=0.4)
                draw = ImageDraw.Draw(image_pil)
                draw = plot_bbox(box, label, draw, id_to_label, color_dict, score)
    else:
        for box, label, score in zip(bboxes, labels, scores):
            if score >= score_threshold:
                draw = plot_bbox(box, label, draw, id_to_label, color_dict, score)
    end_time = rospy.Time.now()
    duration = (end_time - start_time).to_sec()
    print ("Plot duration:", duration)
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
    draw.rectangle([x, y, width, height], outline=label_to_color(label, color_dict), width=5)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', size=20)
    draw.text((x, y), f"{id_to_label[label]}: {score:.2f}" if score is not None else id_to_label[label], font=font, fill=label_to_color(label, color_dict))
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
            0: (204, 255, 204)   # Light Pastel Green
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
        id_to_label = {0: 'Crack'}
                       
    return id_to_label

def quat2rot_matrix(qx, qy, qz, qw):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param q0, q1, q2, q3: qw, qx, qy, qz
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
     
    # First row of the rotation matrix
    r00 = 1 - 2*(qy*qy + qz*qz)
    r01 = 2 * (qx*qy - qw*qz)
    r02 = 2 * (qx*qz + qw*qy)
     
    # Second row of the rotation matrix
    r10 = 2 * (qx*qy + qw*qz)
    r11 = 1 - 2*(qx*qx + qz*qz)
    r12 = 2 * (qy*qz - qw*qx)
     
    # Third row of the rotation matrix
    r20 = 2 * (qx*qz - qw*qy)
    r21 = 2 * (qy*qz + qw*qx)
    r22 = 1 - 2*(qx*qx + qy*qy)
     
    # 3x3 rotation matrix
    rot_matrix = numpy.array([[r00, r01, r02],
                             [r10, r11, r12],
                             [r20, r21, r22]])
                            
    return rot_matrix