#!/usr/bin/env python3
import rospy
#import cv2
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
#from matplotlib import pyplot as plt


def find_indexes(array, x):
    indexes = np.where(array == x)
    return list(zip(indexes[1], indexes[0]))

class CrackMeasurement(): 

    def __init__(self):
        rospy.loginfo("Initializing node!")
        rospy.init_node('crack_localizer_node', anonymous=True, log_level=rospy.DEBUG)
        self.rate = rospy.Rate(20)
        # Testing flags! 
        self.crack_start_reciv = False; self.crack_stop_reciv = False; self.uav_pose_reciv = False; self.last_reciv = 1
        self.pts_ = []; self.e_p = []; self.real_exp = True
        # Test precision 
        self.test_precision = True; 
        # Initialize subscribers and publishers in the end!
        self._init_subscribers()

    def _init_subscribers(self): 
        # Subscribers for the crack_start, crack_end
        self.crack_start_sub = rospy.Subscriber("/crack_start/vrpn_client/estimated_odometry", Odometry, self.crack_start_cb, queue_size=1)
        self.crack_stop_sub = rospy.Subscriber("/crack_end/vrpn_client/estimated_odometry", Odometry, self.crack_stop_cb, queue_size=1)
        self.uav_pose_sub = rospy.Subscriber("/nuckopter/vrpn_client/estimated_odometry", Odometry, self.uav_pose_cb, queue_size=1)
        self.viz_markers_sub = rospy.Subscriber("/viz/markers", MarkerArray, self.marker_array_cb, queue_size=1)

    def marker_array_cb(self, msg): 
        self.pts_ = []
        for marker in msg.markers: 
            self.pts_.append([marker.pose.position.x, marker.pose.position.y, marker.pose.position.z])    

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
        self.last_reciv = rospy.Time.now().to_sec()

    def p2T(self, pose): 
        T = np.matrix((4, 4))
        R = quat2rot_matrix(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        T = np.array([[R[0, 0], R[0, 1], R[0, 2], pose.position.x], 
                     [R[1, 0], R[1, 1], R[1, 2], pose.position.y], 
                     [R[2, 0], R[2, 1], R[2, 2], pose.position.z], 
                     [0, 0, 0, 1]])
        return T

    def run(self): 
        file_path=f"/root/uav_ws/src/mmros_wrapper/scripts/crack_measurement.txt"
        file = open(file_path, 'w')
        while not rospy.is_shutdown(): 
            
            dt = rospy.Time.now().to_sec() - self.last_reciv
            self.reciv_data = self.crack_stop_reciv and self.crack_start_reciv and self.uav_pose_reciv and dt < 0.2
            # Test precision flag to compare crack start and crack stop 
            if self.test_precision == True and self.reciv_data: 
                # Test location of the each crack
                T_w_b = self.p2T(self.uav_pose)
                T_b_c = np.array([[0, 0, 1, 0.045], [-1, 0, 0, 0], [0, -1, 0, 0.096], [0, 0, 0, 1]])
                if len(self.pts_)> 1:  # Found markers in markers
                    p_x = [p[0] for p in self.pts_]; p_y =[p[1] for p in self.pts_]; p_z = [p[2] for p in self.pts_]
                    p_x_ = sum(p_x)/len(p_x); p_y_ = sum(p_y)/len(p_y); p_z_ = sum(p_z)/len(p_z)
                    p = np.array([np.round(p_x_, 3), np.round(p_y_, 3), np.round(p_z_, 3), 1])
                    #rospy.logdebug(f"p_cam_cr: {p}")
                    p_b = np.dot(T_b_c, p); 
                    #rospy.logdebug(f"p_base: {p_b}")
                    p_trans = np.dot(T_w_b, p_b)
                    rospy.logdebug(f"p: {p_trans}")
                    # TODO: Move averaging to utils smhw!
                    p_c = np.array([(self.p_crack_start.x + self.p_crack_stop.x)/2, (self.p_crack_start.y + self.p_crack_stop.y)/2, (self.p_crack_start.z + self.p_crack_stop.z)/2])
                    rospy.logdebug(f"p_c: {p_c}")
                    self.e_p.append([abs(p_trans[0] - p_c[0]), abs(p_trans[1] - p_c[1]), abs(p_trans[2] - p_c[2])])
                    try: 
                        rospy.loginfo("Saving localization!")
                        str_ = f"{p_trans[0]}, {p_trans[1]}, {p_trans[2]}, {p_c[0]}, {p_c[1]}, {p_c[2]}, {T_w_b[0, 3]}, {T_w_b[1, 3]}, {T_w_b[2, 3]}, {p_b[0]}, {p_b[1]}, {p_b[2]} \n"
                        file.write(str_)
                    except Exception as e:
                        print(f"Error writing to {file_path}: {e}")

            if dt > 15.0 and len(self.pts_) > 1 : 
                e_p_ = [sum([i[0] for i in self.e_p])/len(self.e_p), sum([i[1] for i in self.e_p])/len(self.e_p), sum([i[2] for i in self.e_p])/len(self.e_p)]
                rospy.logdebug(f"Average error is: {e_p_}")
                file.close()

            self.rate.sleep()

if __name__ == "__main__": 
    try: 
        cM = CrackMeasurement()
        cM.run()
    except rospy.ROSInterruptException: 
        exit()

