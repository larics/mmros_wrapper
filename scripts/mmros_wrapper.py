#!/home/gideon/archiconda3/envs/mmdeploy/lib/python3.8

import rospy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np

from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
from PIL import Image
import torch

class MMRosWrapper:
    def __init__(self, deploy_cfg_path, model_cfg_path, backend_model_name):
        self.img = None
        self.img_received = False
        self.model_initialized = False
        rospy.init_node('image_subscriber_node', anonymous=True)
        self.rate = rospy.Rate(10)  # 10 Hz, adjust as needed
        self.compr_img_sub = rospy.Subscriber("camera/color/image_raw/compressed", CompressedImage, self.image_callback)
        self.model = self.load_model(deploy_cfg_path, model_cfg_path, backend_model_name)

    def image_callback(self, data):
        try:
            np_arr = np.fromstring(data.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.img = img
            self.img_received = True
        except Exception as e:
            rospy.logerr("Error decoding compressed image: %s", str(e))

    def load_model(self, deploy_cfg_path, model_cfg_path, backend_model_name):
        # Read deploy_cfg and model_cfg
        deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)

        # Replace the following line with the appropriate device selection if needed
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build task and backend model
        task_processor = build_task_processor(model_cfg, deploy_cfg, device)
        model = task_processor.build_backend_model(backend_model_name)
        self.model_initialized = True
	rospy.loginfo("Model succesfully initialized!")

        return model

    def run(self):
        while not rospy.is_shutdown():
            if self.model_initialized and self.img_received:
                img = self.img.copy()

                # Process input image
                input_shape = get_input_shape(deploy_cfg)
                model_inputs, _ = task_processor.create_input(img, input_shape)

                # Capture the start time
                start_time = rospy.Time.now()

                with torch.no_grad():
                    result = model.test_step(model_inputs)

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
            mmWrap = MMRosWrapper(deploy_cfg_path, model_cfg_path, backend_model_name)
            mmWrap.run()
    except rospy.ROSInterruptException:
        pass
