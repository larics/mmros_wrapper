#!/usr/bin/env python3
import rospy
import cv2
import sys
import io
import copy
import numpy as np
import sensor_msgs.point_cloud2 as pc2

from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Pose, Vector3
from mmros_utils.msg import InstSegArray, InstSeg

from utils import ros_image_to_numpy
from matplotlib import pyplot as plt

def find_indexes(array, x):
    indexes = np.where(array == x)
    return list(zip(indexes[1], indexes[0]))

class CrackLocalizer(): 

    def __init__(self):
        rospy.loginfo("Initializing node!")
        rospy.init_node('shm_planner', anonymous=True, log_level=rospy.DEBUG)
        self.rate = rospy.Rate(1)
        self.inst_seg_reciv = False; self.pcl_reciv = False;
        # Initialize subscribers and publishers in the end!
        self._init_publishers()
        self._init_subscribers()

    def _init_subscribers(self): 
        self.inst_seg_sub = rospy.Subscriber("/inst_seg/output", InstSegArray, self.inst_seg_cb, queue_size=1)
        self.pcl_sub = rospy.Subscriber("/camera/depth_registered/points", PointCloud2, self.pcl_cb, queue_size=1)

    def _init_publishers(self): 
        self.viz_pub = rospy.Publisher("/viz/markers", MarkerArray)

    def get_depths(self, pcl, indices, axis="z"):

        # Get current depths from depth cam --> TODO: Change read_points with cam_homography
        depths = pc2.read_points(pcl, [axis], False, uvs=indices)
        return depths

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
        color.r = 0.5; color.g = 0.5; color.b = 0.05; color.a = 0.5; 
        for i, p in enumerate(points): 
            p_ = Pose()
            p_.position.x, p_.position.y, p_.position.z = p[0], p[1], p[2]
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
                mask = self.inst_seg.instances[0].mask
                pts = self.localize_crack(mask)
                viz = True
                if viz:
                    markers = self.visualize_crack_3d(pts)
                    self.viz_pub.publish(markers)
            else: 
                rospy.logwarn(f"Img reciv: {self.inst_seg_reciv}; PCL reciv: {self.pcl_reciv}")
            self.rate.sleep()



if __name__ == "__main__": 
    try: 
        cL = CrackLocalizer()
        cL.run()
    except rospy.ROSInterruptException: 
        exit()