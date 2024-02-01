#!/usr/bin/env python3
import rospy
import cv2
import sys
import io
import copy
import numpy as np
import sensor_msgs.point_cloud2 as pc2

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from mmros_utils.msg import InstSegArray, InstSeg
from visualization.msg import MarkerArray

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

    def localize_crack(mask_msg, sample=1):
        # Get mask
        mask_ = self.inst_seg.instances[0].mask
        np_mask = ros_image_to_numpy(mask_)
        indices = find_indexes(np_mask, 255)
        indices_ = indices[::sample]
        points = list(pc2.read_points(self.pcl, skip_nans=True, uvs=indices, field_names = ("x", "y", "z")))
        return points

    def run(self): 
        self.cnt = 0
        while not rospy.is_shutdown(): 
            if self.inst_seg_reciv and self.pcl_reciv: 
                mask = self.inst_seg.instances[0].mask
                pts = self.localize_crack(mask)
                #print(mask_)
                # Get indices for that mask
                #indices = find_indexes(mask_, 255)
                #rospy.logdebug(f"Indices are: {indices}")
                # Get 3D points for that indices
                #points = get_depths(self.pcl, indices)
                #rospy.logdebug(f"Points are: {points}")
            else: 
                rospy.logwarn(f"Img reciv: {self.inst_seg_reciv}; PCL reciv: {self.pcl_reciv}")
            self.rate.sleep()



if __name__ == "__main__": 
    try: 
        cL = CrackLocalizer()
        cL.run()
    except rospy.ROSInterruptException: 
        exit()