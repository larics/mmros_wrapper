#!/usr/bin/env python3

import rospy
import cv2
import sys
import io
import copy
import torch
import numpy as np
import csv

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from mmros_utils.msg import InstSegArray, InstSeg
from utils import *
from tracker import CentroidTracker
from mmdet.apis import init_detector, inference_detector

class MMRosWrapper:
    def __init__(self, model_cfg_path, checkpoint_file):
        rospy.loginfo("Initializing node!")
        self.img = None
        self.img_received = False
        self.model_initialized = False

        rospy.init_node('mmros_wrapper', anonymous=True, log_level=rospy.INFO)
        self.rate = rospy.Rate(20)  # 10 Hz, adjust as needed
        
        device = 'cuda:0' # or device='cpu'
        self.model = init_detector(model_cfg_path, checkpoint_file, device)
        self.model_initialized = True

        self._init_subscribers()
        self._init_publishers()

        # Simple centroid tracking
        self.tracking = False
        if self.tracking: 
            self.cT = CentroidTracker()
            
        self.anot_type = "taco"
        # Choose net type
        if self.anot_type == "coco": 
            self.color_palette = create_color_palette("coco")
        if self.anot_type == "taco":
            self.color_palette = create_color_palette("taco")
        
        # Model name
        self.model_name = "mask-rcnn-50"

    def _init_subscribers(self): 
        self.compr_img_sub = rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.img_cb, queue_size=1)

    def _init_publishers(self): 
        self.img_pub = rospy.Publisher("/camera/color/image_raw/output", Image, queue_size=1)
        self.mask_pub = rospy.Publisher("/mask/output", Image, queue_size=1)
        self.hsv_mask_pub = rospy.Publisher("/hsv_mask/output", Image, queue_size=1)
        self.inst_seg_pub = rospy.Publisher("/inst_seg/output", InstSegArray, queue_size=1)
        self.compr_img_pub = rospy.Publisher("/camera/color/image_raw/output/compressed", CompressedImage, queue_size=1)

    def img_cb(self, data):
        try:
            np_arr = np.frombuffer(data.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.img = img_rgb
            self.header = data.header
            self.np_arr = np_arr
            self.img_received = True
        except Exception as e: 
            rospy.logerr("Error decoding compressed image: %s", str(e))

        # Convert and publish decompressed message
        debug_img_reciv = False
        if debug_img_reciv:
            img_msg = self.convert_np_array_to_ros_img_msg(data.data, data.header)
            compr_img_msg = CompressedImage()
            compr_img_msg.header = img_msg.header
            compr_img_msg.format = img_msg.format
            compr_img_msg.data = img_msg.data
            self.img_pub.publish(img_msg)
            self.compr_img_pub.publish(compr_img_msg)

    def convert_np_array_to_ros_img_msg(self, data, header):
        pil_img = PILImage.open(io.BytesIO(bytearray(data)))
        img_msg = convert_pil_to_ros_img(pil_img, header)
        return img_msg

    def create_inst_seg_msg(self, labels, masks, scores, num_obj=1): 
        # num_obj argument defines how much instances can we detect
        full_msg = InstSegArray()

        zipped = zip(masks[:num_obj], scores[:num_obj], labels[:num_obj])
        for i, (mask_, score_, label_) in enumerate(zipped):
            inst_msg = InstSeg()
            inst_msg.id = label_
            inst_msg.score = score_
            mask8_ = (mask_ * 255).astype(np.uint8)
            inst_msg.mask = convert_np_array_to_ros_img_msg(mask8_)
            full_msg.instances.append(inst_msg)
        
        rospy.logdebug(f"Number of detected instances is: {i}")
        debug_mask = True
        if debug_mask: 
            self.mask_pub.publish(inst_msg.mask)
        # Publish full instance segmentation msg
        self.inst_seg_pub.publish(full_msg)      

    def extract_results(self, result): 
        labels = result.pred_instances.labels.cpu().detach().numpy()
        bboxes = result.pred_instances.bboxes.cpu().detach().numpy()
        masks = result.pred_instances.masks.cpu().detach().numpy()
        scores = result.pred_instances.scores.cpu().detach().numpy()
        return labels, bboxes, masks, scores

    def track(self, bboxes, scores, plot, track_treshold=0.9): 
        # Convert bboxes to format used in centroid tracker
        rects = filter_bboxes(bboxes, scores, track_treshold)
        # Call update on centroid tracker
        objects = self.cT.update(rects)
        if not plot:
            pil_img = PILImage.fromarray(img)
            draw = ImageDraw.Draw(pil_img)
            for (objectID, centroid) in objects.items():
                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = "ID {}".format(objectID)
                # font = ImageFont.load_default()
                font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', size=20)
                draw.text((centroid[0] - 10, centroid[1] - 10), text, fill=(0, 255, 0), font=font)
                draw.ellipse((centroid[0] - 4, centroid[1] - 4, centroid[0] + 4, centroid[1] + 4), fill=(0, 255, 0))
    
    def mAP_test(self): 
        pass
        # TODO: Add mAP test
        # - Add HSV filtering 
        # - Compare with other image 
        # - Check inference speed 
    
    def run(self):
        # txt Saving durations
        file_path=f"/root/uav_ws/src/mmros_wrapper/scripts/{self.model_name}.txt"
        file = open(file_path, 'w')
        while not rospy.is_shutdown():
            if self.model_initialized and self.img_received:
                header = copy.deepcopy(self.header)
                img = self.img.copy()
                # Capture the start time
                t1 = rospy.Time.now()
                with torch.no_grad():
                    result = inference_detector(self.model, img)
                    labels, bboxes, masks, scores = self.extract_results(result)
                    if self.mAP_test: 
                        self.mAP_test()
                    t2 = rospy.Time.now()
                    dt = (t2 - t1).to_sec()
                    rospy.loginfo("Inference time is: {}".format(dt))
                    try: 
                        file.write(f"{dt}\n")
                    except Exception as e:
                        print(f"Error writing to {file_path}: {e}")
                # Create instance segmentation mask
                self.create_inst_seg_msg(labels, masks, scores)        
                # Test detection
                plot = False
                if plot:    
                    pil_img = plot_result(img, bboxes, labels, scores, 0.3, True, self.anot_type, self.color_palette, masks)
                    ros_img = convert_pil_to_ros_img(pil_img, header)
                    self.img_pub.publish(ros_img)
                # Test tracking
                if self.tracking:
                    self.track(bboxes, scores, plot, track_treshold=0.9)
                # Capture the end time and calculate the duration
                t2 = rospy.Time.now()
                dt = (t2-t1).to_sec()
                rospy.logdebug(f"Duration of model.test_step(model_inputs):{dt}")
            else:
                if not self.model_initialized:
                    rospy.logwarn_throttle(1, "Model not initialized yet.")
                if not self.img_received:
                    rospy.logwarn_throttle(1, "Image not received yet.")

            self.rate.sleep()
        
        file.close()
        
if __name__ == '__main__':
    try:
        if len(sys.argv) < 3:
            print("Usage for checkpoint: rosrun your_package_name MMRosWrapper.py model_cfg_path checkpoint_path")
        else:
            model_cfg_path = sys.argv[1]
            checkpoint_file = sys.argv[2]
            
            print("Model cfg path: {}".format(model_cfg_path))
            print("Checkpoint file: {}".format(checkpoint_file))
                    
            mmWrap = MMRosWrapper(model_cfg_path, checkpoint_file)
            mmWrap.run()
    except rospy.ROSInterruptException:
        exit()