#!/usr/bin/env python3
import rospy
#import cv2
import sys
import io
import copy
import numpy as np
import sensor_msgs.point_cloud2 as pc2

from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Pose, Vector3
from mmros_utils.msg import InstSegArray, InstSeg

from utils import ros_image_to_numpy, convert_np_array_to_ros_img_msg
from matplotlib import pyplot as plt

def find_indexes(array, x):
    indexes = np.where(array == x)
    return list(zip(indexes[1], indexes[0]))

class CrackLocalizer(): 

    def __init__(self):
        rospy.loginfo("Initializing node!")
        rospy.init_node('crack_localizer_node', anonymous=True, log_level=rospy.DEBUG)
        self.rate = rospy.Rate(20)
        self.inst_seg_reciv = False; self.pcl_reciv = False; self.rgb_reciv = False; 
        # Initialize subscribers and publishers in the end!
        self._init_publishers()
        self._init_subscribers()

    def _init_subscribers(self): 
        # TODO: Change queue_size for message synchronizer
        # I don't need compressed for OCTOMAP
        # self.img_sub = rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.img_cb, queue_size=1)
        self.inst_seg_sub = rospy.Subscriber("/inst_seg/output", InstSegArray, self.inst_seg_cb, queue_size=1)
        self.pcl_sub = rospy.Subscriber("/camera/depth_registered/points", PointCloud2, self.pcl_cb, queue_size=1)

    # TODO: Add message synchronizer
    def _init_publishers(self): 
        self.viz_pub = rospy.Publisher("/viz/markers", MarkerArray, queue_size=1)
        self.img_gt_pub = rospy.Publisher("/img_gt", Image, queue_size=1)

    def img_cb(self, msg): 
        #rospy.logdebug("Recived image!")
        self.rgb_reciv = True
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception as e: 
            rospy.logerr("Error decoding compressed image: %s", str(e))
        # Apply thresholding --> Add to utils method! 
        threshold_value = 180
        max_value = 255
        _, binary_image = cv2.threshold(bw_img, threshold_value, max_value, cv2.THRESH_BINARY)
        img_gt = convert_np_array_to_ros_img_msg(binary_image)
        self.img_gt_pub.publish(img_gt)

    def inst_seg_cb(self, msg): 
        self.inst_seg_reciv = True
        self.inst_seg = msg

    def pcl_cb(self, msg): 
        self.pcl_reciv = True
        self.pcl = msg

    def localize_crack(self, mask_msg, sample=1):
        # Get mask
        mask_ = self.inst_seg.instances[0].mask
        np_mask = ros_image_to_numpy(mask_)
        indices = find_indexes(np_mask, 255)
        indices_ = indices[::sample]
        points = list(pc2.read_points(self.pcl, skip_nans=True, uvs=indices, field_names = ("x", "y", "z")))
        return points

    def visualize_crack_3d(self, points):
        markers = MarkerArray()
        color = ColorRGBA()
        color.r = 1.0; color.g = 0.0; color.b = 0.00; color.a = 1.0; 
        for i, p in enumerate(points): 
            p_ = Pose()
            p_.position.x, p_.position.y, p_.position.z = p[0], p[1], p[2]
            p_.orientation.x = 0.0; p_.orientation.y = 0.0; p_.orientation.z = 0.0; p_.orientation.w = 1.0; 
            s = Vector3(0.01, 0.01, 0.01)
            # http://docs.ros.org/en/noetic/api/visualization_msgs/html/msg/Marker.html
            marker_ = Marker()
            marker_.header.frame_id = "map"
            marker_.id = i
            marker_.action = 0; 
            marker_.type = 1;  # 1 CUBE, 2 SPHERE
            marker_.pose = p_
            marker_.scale = s
            marker_.color = color
            markers.markers.append(marker_)
        return markers

    def run(self): 
        self.cnt = 0
        while not rospy.is_shutdown(): 
            if self.inst_seg_reciv and self.pcl_reciv: 
                # for now, detect only one instance
                mask = self.inst_seg.instances[0].mask
                # TODO: Add comparison with thresholding or smthing like that 
                pts = self.localize_crack(mask, 20)
                viz = True
                if viz:
                    markers = self.visualize_crack_3d(pts)
                    self.viz_pub.publish(markers)
            else: 
                rospy.logwarn_throttle(2, f"Img reciv: {self.inst_seg_reciv}; PCL reciv: {self.pcl_reciv}")
            self.rate.sleep()

if __name__ == "__main__": 
    try: 
        cL = CrackLocalizer()
        cL.run()
    except rospy.ROSInterruptException: 
        exit()
