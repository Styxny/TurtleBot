#! /usr/bin/env python


# Python libs
import sys, time

# numpy and scipy
import numpy as np
# from scipy.ndimage import filters
from std_msgs.msg import String

# OpenCV
import cv2

# Ros libraries
import roslib
import rospy

# Ros Messages
from sensor_msgs.msg import CompressedImage


def callback(ros_data):
    '''Callback function of subscribed topic.
    Here images get converted and features detected'''
    # if VERBOSE:
    #     print
    #     'received image of type: "%s"' % ros_data.format

    # Publisher
    pub = rospy.Publisher('circle_center', String, queue_size=10)
    r = rospy.Rate(10)

    #### direct conversion to CV2 ####
    np_arr = np.fromstring(ros_data.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    #Image Size
    height, width = image_np.shape[:2]
    center = width/2
    # print("the center is:", center)

    kernel = np.ones((5, 5), np.uint8)
    blur = cv2.blur(image_np,(5,5))
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    normalizedimg = np.zeros((500,500))
    norm_img = cv2.normalize(gray,normalizedimg,0,255, cv2.NORM_MINMAX)


    # Find Circles
    circles = cv2.HoughCircles(norm_img, cv2.HOUGH_GRADIENT, 1, 90, param1=70, param2=60, minRadius=0, maxRadius=0)

    msg = 'none'

    if circles is not None:
        circle_center = circles[0][0][0]
        if circle_center < center:
            msg = "left"
            print(msg)
        elif circle_center > center:
            msg = "right"
            print(msg)

    # publish the velocity
    # rospy.loginfo(msg)
    pub.publish(msg)
    # wait for 0.1 seconds (10 HZ) and publish again
    # r.sleep()

    #show image
    # cv2.imshow('ros_image', image_np)
    # cv2.waitKey(2)
    # cv2.imshow('noramlized_Image', norm_img)


def center_finder():

    # Initialize node
    rospy.init_node('find_ball')
    r = rospy.Rate(10)

    #Capture frame-by-frame
    ros_image = rospy.Subscriber("/raspicam_node/image/compressed", CompressedImage, callback, queue_size = 1)



if __name__ == '__main__':
    center_finder()
    try:
        rospy.spin()
    except rospy.KeyboardInterrupt:
        print('Shutting Down Image Subscriber')
    cv2.destroyAllWindows()