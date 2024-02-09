#!/usr/bin/env python3
import rospy
import cv2
import sys
import io
import copy
import numpy as np
import sensor_msgs.point_cloud2 as pc2
# TF is compiled for python2
# How to compile TF for python3: https://answers.ros.org/question/326226/importerror-dynamic-module-does-not-define-module-export-function-pyinit__tf2/
#import tf.transformations as tr

from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Pose, Vector3
from mmros_utils.msg import InstSegArray, InstSeg
from nav_msgs.msg import Odometry

from utils import ros_image_to_numpy, convert_np_array_to_ros_img_msg, quat2rot_matrix
from matplotlib import pyplot as plt


def find_indexes(array, x):
    indexes = np.where(array == x)
    return list(zip(indexes[1], indexes[0]))

class CrackLocalizer(): 

    def __init__(self):
        rospy.loginfo("Initializing node!")
        rospy.init_node('crack_localizer_node', anonymous=True, log_level=rospy.DEBUG)
        self.rate = rospy.Rate(20)
        # Entering conditions
        self.inst_seg_reciv = False; self.pcl_reciv = False; self.rgb_reciv = False; 
        self.crack_start_reciv = False; self.crack_stop_reciv = False; self.uav_pose_reciv = False; 
        self.pts_ = []; 
        # Test precision 
        self.test_precision = True; 
        # Initialize subscribers and publishers in the end!
        self._init_publishers()
        self._init_subscribers()

    def _init_subscribers(self): 
        # TODO: Change queue_size for message synchronizer
        self.img_sub = rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.img_cb, queue_size=1)
        self.inst_seg_sub = rospy.Subscriber("/inst_seg/output", InstSegArray, self.inst_seg_cb, queue_size=1)
        self.pcl_sub = rospy.Subscriber("/camera/depth_registered/points", PointCloud2, self.pcl_cb, queue_size=1)

        # Subscribers for the crack_start, crack_end
        self.crack_start_sub = rospy.Subscriber("/crack_start/vrpn_client/estimated_odometry", Odometry, self.crack_start_cb, queue_size=1)
        self.crack_stop_sub = rospy.Subscriber("/crack_end/vrpn_client/estimated_odometry", Odometry, self.crack_stop_cb, queue_size=1)
        self.uav_pose_sub = rospy.Subscriber("/nuckopter/vrpn_client/estimated_odometry", Odometry, self.uav_pose_cb, queue_size=1)
        self.viz_markers_sub = rospy.Subscriber("/viz/markers", MarkerArray, self.marker_array_cb, queue_size=1)

    # TODO: Add message synchronizer
    def _init_publishers(self): 
        self.viz_pub = rospy.Publisher("/viz/markers", MarkerArray, queue_size=1)
        self.img_gt_pub = rospy.Publisher("/img_gt", Image, queue_size=1)

    # Callbacks for the gt measurement 
    def crack_start_cb(self, msg): 
        self.crack_start_reciv = True
        self.p_crack_start = Vector3()
        self.p_crack_start.x = msg.pose.pose.position.x
        self.p_crack_start.y = msg.pose.pose.position.y
        self.p_crack_start.z = msg.pose.pose.position.z

    def crack_stop_cb(self, msg): 
        self.crack_stop_reciv = True
        self.p_crack_stop = Vector3()
        self.p_crack_stop.x = msg.pose.pose.position.x
        self.p_crack_stop.y = msg.pose.pose.position.y
        self.p_crack_stop.z = msg.pose.pose.position.z
    
    def uav_pose_cb(self, msg): 
        self.uav_pose_reciv = True
        self.uav_pose = Pose()
        self.uav_pose.position.x = msg.pose.pose.position.x
        self.uav_pose.position.y = msg.pose.pose.position.y
        self.uav_pose.position.z = msg.pose.pose.position.z
        self.uav_pose.orientation.x = msg.pose.pose.orientation.x
        self.uav_pose.orientation.y = msg.pose.pose.orientation.y
        self.uav_pose.orientation.z = msg.pose.pose.orientation.z
        self.uav_pose.orientation.w = msg.pose.pose.orientation.w

    def marker_array_cb(self, msg): 
        self.pts_ = []
        for marker in msg.markers: 
            self.pts_.append([marker.pose.position.x, marker.pose.position.y, marker.pose.position.z])

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

    def p2T(self, pose): 
        T = np.matrix((4, 4))
        R = quat2rot_matrix(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z)
        T = np.array([[R[0, 0], R[0, 1], R[0, 2], pose.position.x], 
                     [R[1, 0], R[1, 1], R[1, 2], pose.position.y], 
                     [R[2, 0], R[2, 1], R[2, 2], pose.position.z], 
                     [0, 0, 0, 1]])
        return T

    def hsv_filtering(self, img, hue = [0, 255], saturation = [0, 255], value=[0, 255]):
        # Convert the ROS image message to an OpenCV image
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # Convert the image from BGR to HSV color space
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Define the range of the color you want to filter in HSV
        lower_range = np.array([hue[0], saturation[0], value_min[0]])
        upper_range = np.array([hue[1], saturation[1], value_max[1]])
        # Create a mask for the specified HSV range
        mask = cv2.inRange(hsv_img, lower_range, upper_range)
        # Apply the mask to the original image
        filtered_img = cv2.bitwise_and(img, img, mask=mask)

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

            self.reciv_data = self.crack_stop_reciv and self.crack_start_reciv and self.uav_pose_reciv
            # Test precision flag to compare crack start and crack stop 
            if self.test_precision == True and self.reciv_data: 
                # Test location of the each crack
                rospy.logdebug("UAV pose is: {}".format(self.uav_pose))
                T_w_b = self.p2T(self.uav_pose)
                T_b_c = np.array([[0, 0, -1, 0.045], [0, -1, 0, 0, ], [1, 0, 0, 0.096], [0, 0, 0, 1]])
                rospy.logdebug(f"T is: {T_w_b}")
                if len(self.pts_)> 1:  # Found markers in markers
                    p_x = [p[0] for p in self.pts_]; p_y =[p[1] for p in self.pts_]; p_z = [p[2] for p in self.pts_]
                    p_x_ = sum(p_x)/len(p_x); p_y_ = sum(p_y)/len(p_y); p_z_ = sum(p_z)/len(p_z)
                    p = np.array([p_x, p_y, p_z, 1])
                    p_trans = np.dot(T_w_b, np.dot(T_b_c, p))
                    rospy.logdebug(f"p_trans: {p_trans}") 
                    rospy.logdebug(f"p_crack_mid: {}")

                # TODO: Maybe add direct comparison to pts estimated from camera? :) But I can sort it out with markers
            else: 
                rospy.logwarn_throttle(2, f"Img reciv: {self.inst_seg_reciv}; PCL reciv: {self.pcl_reciv}")
            self.rate.sleep()



if __name__ == "__main__": 
    try: 
        cL = CrackLocalizer()
        cL.run()
    except rospy.ROSInterruptException: 
        exit()