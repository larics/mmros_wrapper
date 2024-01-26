#!/usr/bin/env python3
import rospy
import cv2
import sys
import io
import copy

from mmros_utils.msg import InstSegArray, InstSeg



class Extractor(): 

    def __init__(self):
        rospy.loginfo("Initializing node!")
        rospy.init_node('shm_planner', anonymous=True)
        self.rate = rospy.Rate(20)
        self.inst_seg_reciv = False
        # Initialize subscribers and publishers in the end!
        self._init_subscribers()

    def _init_subscribers(self): 
        self.inst_seg_sub = rospy.Subscriber("/inst_seg/output", InstSegArray, self.inst_seg_cb, queue_size=1)

    def _init_publishers(self): 
        pass

    def inst_seg_cb(self, msg): 

        self.inst_seg_reciv = True
        self.inst_seg = msg

    def run(self): 
        while not rospy.is_shutdown(): 
            if self.inst_seg_reciv: 
                print("Number of detected masks: {}".format(len(self.inst_seg.instances)))
            self.rate.sleep()


if __name__ == "__main__": 
    try: 
        ext = Extractor()
        ext.run()
    except rospy.ROSInterruptException: 
        exit()

