#!/usr/bin/env python

import rospy
# from sensor_msgs.msg import CompressedImage
import cv2
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import numpy as np
import time
import math

global pub
found = False



def callback_chase(pt):
    global point
    global found

    point = pt

    found = True

def Init():
    global pub

    # Define Publisher
    pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

    # Subscribe to detectObject
    rospy.Subscriber("thetaDist", Point, callback_chase)

    # Initialize node
    rospy.init_node('chaseObject', anonymous=True)

if __name__ == "__main__":
    try:
        Init()
    except rospy.ROSInterruptException:
        pass
rate = rospy.Rate(10)

msg = Twist()

msg.linear.x = 0
msg.linear.y = 0
msg.linear.z = 0
msg.angular.x = 0
msg.angular.y = 0
msg.angular.z = 0

while not rospy.is_shutdown():
    # print(found)

    if found:
        if point.x != 999:

            range = 0.4-point.x

            theta = np.deg2rad(point.y)

            # print("theta before:", theta)

            if theta > np.pi:
                theta = theta - 2*np.pi

            # print("theta after:", theta)

            msg.linear.x = -0.3 * range
            msg.angular.z = 1.6 * theta

            # rospy.loginfo(msg)
            pub.publish(msg)
            rate.sleep()
        else:
            msg.linear.x = 0
            msg.linear.y = 0
            msg.linear.z = 0
            msg.angular.x = 0
            msg.angular.y = 0
            msg.angular.z = 0
            pub.publish(msg)

