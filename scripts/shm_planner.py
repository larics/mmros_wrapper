#!/usr/bin/env python3
import rospy
import cv2
import sys
import io
import copy
import numpy as np

from mmros_utils.msg import InstSegArray, InstSeg
from utils import ros_image_to_numpy
from matplotlib import pyplot as plt

def show_imgs(edges, pts): 
    # Create a figure and axis
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(10)
    fig.set_figwidth(10)
    # Plot the first horizontal subplot
    ax1.imshow(edges, cmap='gray')
    ax1.set_title('Edges image')

    # Plot the second horizontal subplot
    ax2.scatter(pts[:, 0], pts[:, 1], color='green')
    #ax2.set_xlim([0, 1000]); ax2.set_ylim([0, 600])

    # Set common labels and show the figure
    plt.tight_layout()  # Adjust layout to prevent clipping of titles
    plt.show()

##  OR SELF!
def hough_lines_p(edges, dividor, show=True): 
    num_votes = int(min(edges.shape)/dividor)
    minLineLength = 30; maxLineGap = 5
    lines = cv2.HoughLinesP(edges,cv2.HOUGH_PROBABILISTIC, np.pi/180, num_votes, minLineLength, maxLineGap)
    print(f"Number of detected lines for {num_votes} votes is: {len(lines)}")
    pts_ = None
    for x in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[x]:
            p1 = np.array([x1, y1]); p2 = np.array([x2, y2])
            if x == 0:
                pts_ = np.vstack((p1, p2))
            else: 
                pts_ = np.vstack((pts_, p1, p2))
            #cv2.line(inputImage,(x1,y1),(x2,y2),(0,128,0),2, cv2.LINE_AA)
            pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
            #cv2.polylines(edges, [pts_], True, (255,255,255), 3)
    if show:  
        show_imgs(edges, pts_)

    return pts

def sortxy(pts):
    x = pts[:, 1]; y = pts[:, 0]
    # Create a list of tuples (value, index)
    sort_x = list(zip(x, range(len(x))))
    # Sort the list of tuples based on values
    sort_xix = sorted(sort_x, key=lambda x: x[0])
    # Unzip the sorted tuples
    x, ix = zip(*sort_xix)
    # Use the sorted indexes to reorder the other list
    sort_y = [y[i] for i in ix]
    return x, sort_y

def polynomial_fitting(pts, order): 
    start_time = time.time()
    # TODO: Speed up by sorting just one time :) 
    x, y = sortxy(pts)
    # Get polynomes
    p = np.polyfit(x, y, order); f = np.poly1d(p)
    x_new = np.linspace(min(x), max(x), num=len(list(x))); y_new = f(x_new)
    e = get_error(y, y_new)
    end_time = time.time()
    t = end_time - start_time
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y)
    plt.plot(x_new, y_new)
    plt.title(f"Poly of order {order} with error {round(e,1)}")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid()
    return y, y_new

def get_error(y, y_new): 
    return np.sqrt(np.sum((y-y_new)**2))

def fit_multiple_orders(pts, num_orders):
    orders = list(range(0, num_orders))
    start_t = time.time()
    for o in orders:
        y, y_new = polynomial_fitting(pts, o)
        check_error(y, y_new)
        end_t = time.time()

class SHMPlanner(): 

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

    def create_trajectory(self, instance_mask): 
        
        d = 10
        # Transform instance to the numpy array
        instance_mask = ros_image_to_numpy(instance_mask)
        pts = hough_lines_p(instance_mask, d)


    def run(self): 
        while not rospy.is_shutdown(): 
            if self.inst_seg_reciv: 
                # mask is sensor_msgs/Image
                instance = self.inst_seg.instances[0].mask
                trajectory = self.create_trajectory(instance)
            self.rate.sleep()


if __name__ == "__main__": 
    try: 
        shmp = SHMPlanner()
        shmp.run()
    except rospy.ROSInterruptException: 
        exit()

