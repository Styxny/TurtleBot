#!/usr/bin/env python
"""
For lab4 in AE-7785 Intro to Robotics

Author: Jacob Stickney
"""
import rospy
from sensor_msgs.msg import LaserScan
import cv2
from geometry_msgs.msg import Point
from std_msgs.msg import String
import numpy as np


def close_object(ranges):
    ind = []
    counts = []

    #What get index values for the ranges less than 0.5
    for n in range(len(ranges)):
        if ranges[n] < 0.4:
            ind.append(n)

    #For each index, how many more indexes within +- 4 of it (try to filter out error values or random mis-measurements)
    for n in ind:
        count = 0
        for i in ind:
            if (i <= (n+4)) and (i >= (n-4)):
                count += 1

        counts.append(count)

    # Check if counts is empty else return none
    if counts != []:
        # Get the max counts from the list
        max_counts = max(counts)
        print(max_counts)

        if max_counts <= 2:
            # print("I see nothing")
            obs_ind = None
            return obs_ind

        # Get the index in count list for max_counts
        max_ind = counts.index(max_counts)

        # Get the index for the ranges of the obstacle
        obs_ind = ind[max_ind]

    else:
        obs_ind = None

    return obs_ind

# Subscribe to both lidar and detect object and determine angular position and distance to object
def callback_lidar(data):
    global pub

    range_max = data.range_max
    range_min = data.range_min

    #Initialize left and right side data
    left_side = []
    right_side = []

    #Get range values from for 45deg left of center and 45deg right of center
    left_side = list(data.ranges[:45])
    right_side = list(data.ranges[-45:])

    #Flip order of left data, (current order 0 -> 45deg, want to make the 45deg the new zero and 0deg the new 45deg)
    left_side.reverse()

    #Flip order of right data for same reason as left
    right_side.reverse()

    #Append range data together to get 90 degrees worth of data from with far left being 0deg and far right being 90deg
    all_ranges = left_side + right_side

    # If lidar doesn't see anything, make the range the max distance
    for i in range(len(all_ranges)):
        if all_ranges[i] < range_min:
            all_ranges[i] = range_max

    # Check if objects are close
    obs_ind = close_object(all_ranges)

    # Initiailize Message
    msg = Point()

    if obs_ind != None:
        msg.x = all_ranges[obs_ind]
        msg.y = obs_ind
        rospy.loginfo(msg)
        pub.publish(msg)

    else:
        msg.x = 999
        msg.y = 999
        rospy.loginfo(msg)
        pub.publish(msg)

def Init():

    global pub

    # Define Publisher
    pub = rospy.Publisher('obstacle', Point, queue_size=1)

    # Subscribe to Lidar information
    rospy.Subscriber('scan', LaserScan, callback_lidar)

    # Initialize node
    rospy.init_node('object_range', anonymous=True)

    # Spin
    rospy.spin()

if __name__ == '__main__':
    try:
        Init()
    except rospy.ROSInterruptException:
        pass

