#! /root/archiconda3/envs/mmdeploy/bin/python
import rospy
import cv2
import sys
import io
import copy


class Extractor(): 

    def __init__(self):
        rospy.loginfo("Initializing node!")
        rospy.init_node('extractor_node', anonymous=True)
        self.rate = rospy.Rate(20)

    def _init_subscribers(self): 
        self.inst_seg_sub = rospy.Subscriber("/inst_seg", CompressedImage, self.inst_seg_cb, queue_size=1)

    def _init_publishers(self): 
        pass

    def inst_seg_cb(self, msg): 

        self.mask = msg.InstSegArray[0].mask

    def run(): 
        while not rospy.is_shutdown(): 
            print("Running")
            self.rate.sleep()


if __name__ == "__main__": 
    try: 
        ext = Extractor()
        ext.run()
    except rospy.ROSInterruptException: 
        exit()

