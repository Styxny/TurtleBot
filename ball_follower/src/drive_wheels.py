#! /usr/bin/env python

import numpy as np
import cv2
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist


def callback(data):

    pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 10)
    rate = rospy.Rate(10)

    msg = Twist()



    mess = str(data.data)

    print(mess)
    # If center on right turn right, elseif on left turn left, else do nothing
    if mess == 'left':
        msg.angular.z = 0.5
    elif mess == 'right':
        msg.angular.z = -0.5

    # while not rospy.is_shutdown():
    # publish the velocity
    # rospy.loginfo(msg)
    pub.publish(msg)
    # wait for 0.1 seconds (10 HZ) and publish again
    # rate.sleep()

def turn():

    rospy.init_node('drive_wheels')
    Sub = rospy.Subscriber("circle_center", String, callback)


if __name__ == "__main__":
    turn()
    try:
        rospy.spin()
    except rospy.KeyboardInterrupt:
        print('Shutting Down Turn')