#! /home/gideon/archiconda3/envs/mmdeploy/bin/python3.8
import rospy
import cv2
import sys
import io
import copy
import torch
import numpy as np

from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from utils import *
from tracker import CentroidTracker

class MMRosWrapper:
    def __init__(self, deploy_cfg_path, model_cfg_path, backend_model_name):
        rospy.loginfo("Initializing node!")
        self.img = None
        self.img_received = False
        self.model_initialized = False
        rospy.init_node('image_subscriber_node', anonymous=True)
        self.rate = rospy.Rate(10)  # 10 Hz, adjust as needed
        self.compr_img_sub = rospy.Subscriber("camera/color/image_raw/compressed", CompressedImage, self.image_callback, queue_size=1)
        self.img_pub = rospy.Publisher("camera/color/image_raw/decompressed", Image, queue_size=10)
        self.model = self.load_model(deploy_cfg_path, model_cfg_path, backend_model_name)
        self.tracking = True
        if self.tracking: 
            self.cT = CentroidTracker()

    def image_callback(self, data):
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
            self.img_pub.publish(img_msg)

    def convert_np_array_to_ros_img_msg(self, data, header):
        pil_img = PILImage.open(io.BytesIO(bytearray(data)))
        img_msg = convert_pil_to_ros_img(pil_img, header)
        return img_msg

    def load_model(self, deploy_cfg_path, model_cfg_path, backend_model_name):
        # Read deploy_cfg and model_cfg
        self.deploy_cfg, self.model_cfg = load_config(deploy_cfg_path, model_cfg_path)
        debug_pths=False
        if debug_pths:
            print("deploy_cfg: {}".format(self.deploy_cfg))
            print("model_cfg: {}".format(self.model_cfg))
        # jetson config --> cuda:0
        device = "cuda:0"

        # Build task and backend model
        self.task_processor = build_task_processor(self.model_cfg, self.deploy_cfg, device)
        model = self.task_processor.build_backend_model([backend_model_name])
        self.model_initialized = True
        rospy.loginfo("Model succesfully initialized!")

        return model

    def run(self):
        while not rospy.is_shutdown():
            if self.model_initialized and self.img_received:
                header = copy.deepcopy(self.header)
                img = self.img.copy()

                # Process input image
                input_shape = get_input_shape(self.deploy_cfg)
                model_inputs, _ = self.task_processor.create_input(img, input_shape)

                # Capture the start time
                start_time = rospy.Time.now()

                with torch.no_grad():
                    result = self.model.test_step(model_inputs)
                    labels = result[0].pred_instances.labels.cpu().detach().numpy()
                    bboxes = result[0].pred_instances.bboxes.cpu().detach().numpy()
                    masks = result[0].pred_instances.masks.cpu().detach().numpy()
                    scores = result[0].pred_instances.scores.cpu().detach().numpy()
                    
                # Test detection
                plot = True
                if plot:    
                    pil_img = plot_result(img, bboxes, labels, scores, 0.25, True, masks)
                    #pil_img = plot_masks(pil_img, masks, labels, scores) 
                
                # Test tracking
                if self.tracking:
                    # Convert bboxes to format used in centroid tracker
                    rects = filter_bboxes(bboxes, scores)
                    # Call update on centroid tracker
                    objects = self.cT.update(rects)
                    if not plot:
                        pil_img = PILImage.fromarray(img)
                    draw = ImageDraw.Draw(pil_img)
                    for (objectID, centroid) in objects.items():
                        # draw both the ID of the object and the centroid of the
                        # object on the output frame
                        text = "ID {}".format(objectID)
                        font = ImageFont.load_default()
                        draw.text((centroid[0] - 10, centroid[1] - 10), text, fill=(0, 255, 0), font=font)
                        draw.ellipse((centroid[0] - 4, centroid[1] - 4, centroid[0] + 4, centroid[1] + 4), fill=(0, 255, 0))
                    
                # Publish plotted img
                ros_img = convert_pil_to_ros_img(pil_img, header)
                self.img_pub.publish(ros_img)
                
                # Capture the end time and calculate the duration
                end_time = rospy.Time.now()
                duration = (end_time - start_time).to_sec()

                print("Duration of model.test_step(model_inputs):", duration)
            else:
                if not self.model_initialized:
                    rospy.logwarn("Model not initialized yet.")
                if not self.img_received:
                    rospy.logwarn("Image not received yet.")

            self.rate.sleep()

if __name__ == '__main__':
    try:
        if len(sys.argv) < 4:
            print("Usage: rosrun your_package_name MMRosWrapper.py deploy_cfg_path model_cfg_path backend_model_name")
        else:
            deploy_cfg_path = sys.argv[1]
            model_cfg_path = sys.argv[2]
            backend_model_name = sys.argv[3]
            print("Deploy cfg path: {}".format(deploy_cfg_path))
            print("Model cfg path: {}".format(model_cfg_path))
            print("Backend model name: {}".format(backend_model_name))
            mmWrap = MMRosWrapper(deploy_cfg_path, model_cfg_path, backend_model_name)
            mmWrap.run()
    except rospy.ROSInterruptException:
        pass
