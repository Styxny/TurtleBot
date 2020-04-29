#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from std_msgs.msg import String
import numpy as np

image = False
lidar = False

# Subscribe to both lidar and detect object and determine angular position and distance to object
def callback_lidar(data):

    global ldata   # ball_range will be global
    global lidar

    ldata = data

    lidar = True
    # print('hello lidar')

def callback_detect(ball_center):

    global theta_ball
    global image


    deg_pix = 62.2/360  # how many degrees per pixel

    if ball_center.x != 999:
        ball_loc = ball_center.x - 180  # number of pixels Ball center left or right of center
        theta_ball = deg_pix * ball_loc   # Degrees to ball center from camera center

        if theta_ball <0:
            theta_ball = -1 * theta_ball
        elif theta_ball>0:
            theta_ball = 360 - theta_ball

        theta_ball = np.rint(theta_ball)     #Round value to nearest integer
        theta_ball = np.int(theta_ball)

    else:
        theta_ball = 999

    image = True
    # print('hello detection')

def Init():

    global pub

    # Define Publisher
    pub = rospy.Publisher('thetaDist', Point, queue_size=1)

    # Subscribe to detectObject
    rospy.Subscriber("imageLocation", Point, callback_detect)

    # Subscribe to Lidar information
    rospy.Subscriber("scan", LaserScan, callback_lidar)

    # Initialize node
    rospy.init_node('object_range', anonymous=True)


if __name__ == '__main__':
    try:
        Init()
    except rospy.ROSInterruptException:
        pass

rate = rospy.Rate(10)
print("image:", image)
print("lidar:", lidar)

while not rospy.is_shutdown():
    if image:
        # print("theta_ball:", theta_ball)

        if lidar:
            if theta_ball != 999:
                theta_ball = theta_ball%360
                # theta_ball_min = (theta_ball -1)%360
                # theta_ball_max = (theta_ball+1)%360
                ball_range = ldata.ranges[theta_ball]  # ball distance in m
                # ball_range_min = ldata.ranges[theta_ball_min]
                # ball_range_max = ldata.ranges[theta_ball_max]
                # ball_range = (1/3)*(ball_range + ball_range_max + ball_range_min)

                # if ball_range < ldata.range_min:
                #     ball_range = 0.120
                # elif ball_range > ldata.range_max:
                #     ball_range = 3.5

                # print("range:", ball_range)

                msg = Point()
                msg.x = ball_range
                msg.y = theta_ball
                # rospy.loginfo(msg)
                pub.publish(msg)

            else:
                msg = Point()
                msg.x=999
                msg.y=999
                # rospy.loginfo(msg)
                pub.publish(msg)

        rate.sleep()