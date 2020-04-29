#!/usr/bin/env python
"""
For lab4 in AE-7785 Intro to Robotics

Author: Jacob Stickney
"""
import rospy
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from tf.msg import *
import math
import numpy as np
import cv2
import csv

point = None
frontRange = None
camImage = None
counter = 0
enum = 0    # counter to remember what I was doing previously

location = None
start = False

#PID controller for Robot
class PID(object):
    """ A basic pid class.

    This class implements a generic structure that can be used to
    create a wide range of pid controllers. It can function
    independently or be subclassed to provide more specific controls
    based on a particular control loop.

    In particular, this class implements the standard pid equation:

    $command = -p_{term} - i_{term} - d_{term} $

    where:

    $ p_{term} = p_{gain} * p_{error} $
    $ i_{term} = i_{gain} * i_{error} $
    $ d_{term} = d_{gain} * d_{error} $
    $ i_{error} = i_{error} + p_{error} * dt $
    $ d_{error} = (p_{error} - p_{error last}) / dt $

    given:

    $ p_{error} = p_{state} - p_{target} $.
    """

    def __init__(self, p_gain, i_gain, i_max, i_min):
        """Constructor, zeros out Pid values when created and
        initialize Pid-gains and integral term limits.

        Parameters:
          p_gain     The proportional gain.
          i_gain     The integral gain.
          i_max      The integral upper limit.
          i_min      The integral lower limit.
        """
        self._p_gain = p_gain
        self._i_gain = i_gain
        self._i_max = i_max
        self._i_min = i_min
        self.reset()

    def reset(self):
        """  Reset the state of this PID controller """
        self._p_error_last = 0.0  # Save position state for derivative
        # state calculation.
        self._p_error = 0.0  # Position error.
        self._d_error = 0.0  # Derivative error.
        self._i_error = 0.0  # Integator error.
        self._cmd = 0.0  # Command to send.
        self._last_time = None  # Used for automatic calculation of dt.

    @property
    def p_gain(self):
        """ Read-only access to p_gain. """
        return self._p_gain

    @property
    def i_gain(self):
        """ Read-only access to i_gain. """
        return self._i_gain


    @property
    def i_max(self):
        """ Read-only access to i_max. """
        return self._i_max

    @property
    def i_min(self):
        """ Read-only access to i_min. """
        return self._i_min

    @property
    def p_error(self):
        """ Read-only access to p_error. """
        return self._p_error

    @property
    def i_error(self):
        """ Read-only access to i_error. """
        return self._i_error

    # @property
    # def d_error(self):
    #     """ Read-only access to d_error. """
    #     return self._d_error

    @property
    def cmd(self):
        """ Read-only access to the latest command. """
        return self._cmd

    def __str__(self):
        """ String representation of the current state of the controller. """
        result = ""
        result += "p_gain:  " + str(self.p_gain) + "\n"
        result += "i_gain:  " + str(self.i_gain) + "\n"
        result += "i_max:   " + str(self.i_max) + "\n"
        result += "i_min:   " + str(self.i_min) + "\n"
        result += "p_error: " + str(self.p_error) + "\n"
        result += "i_error: " + str(self.i_error) + "\n"
        # result += "d_error: " + str(self.d_error) + "\n"
        result += "cmd:     " + str(self.cmd) + "\n"
        return result

    def update_PID(self, p_error, dt, clear):
        """  Update the Pid loop with nonuniform time step size.

        Parameters:
          p_error  Error since last call (p_state - p_target)
          dt       Change in time since last call, in seconds, or None.
                   If dt is None, then the system clock will be used to
                   calculate the time since the last update.
        """
        if clear == 1:
            self._i_error = 0

        # if dt == None:
        #     cur_time = time.time()
        #     if self._last_time is None:
        #         self._last_time = cur_time
        #     dt = cur_time - self._last_time
        #     self._last_time = cur_time

        self._p_error = p_error  # this is pError = pState-pTarget
        if dt == 0 or math.isnan(dt) or math.isinf(dt):
            return 0.0

        # Calculate proportional contribution to command
        p_term = self._p_gain * self._p_error

        # Calculate the integral error
        self._i_error += dt * self._p_error

        # Calculate integral contribution to command
        i_term = self._i_gain * self._i_error

        # Limit i_term so that the limit is meaningful in the output
        if i_term > self._i_max and self._i_gain != 0:
            i_term = self._i_max
            self._i_error = i_term / self._i_gain
        elif i_term < self._i_min and self._i_gain != 0:
            i_term = self._i_min
            self._i_error = i_term / self._i_gain

        # Calculate the derivative error
        # self._d_error = (self._p_error - self._p_error_last) / dt
        self._p_error_last = self._p_error

        # Calculate derivative contribution to command
        # d_term = self._d_gain * self._d_error

        # self._cmd = -p_term - i_term - d_term
        self._cmd = p_term + i_term

        return self._cmd

# Robot Class
class Robot:

    def __init__(self):
        global pub
        self.Init = True
        # Initialize the PID controllers
        self.linear_control = PID(0.1, 0.01, 0.1, -0.1)
        self.angular_control = PID(0.28, 0.01, 0.1, -0.1)

        # Distance to switch from follow to classify
        self.switch_dis = 0.35

        # If using list of goal points
        # self.goals = goals
        self.enum = 0

        # Initialize all the odometry position
        self.Init_ang = 0
        self.Init_pos_x = 0
        self.Init_pos_y = 0
        self.Init_pos_z = 0
        self.globalPos_x = 0
        self.globalPos_y = 0
        self.globalAng = 0

        # Remeber what I was doing last
        self.lastActivity = None

        # What is my x,y location when I switched from one action to another
        self.switch_loc = None
        self.desired_heading = None

        # Initialize KNN to use for sign recognition
        self.knn = self.trainKNN()

    def update_Odometry(self,Odom):
        position = Odom.pose.pose.position

        #Orientation uses the quaternion aprametrization.
        #To get the angular position along the z-axis, the following equation is required.
        q = Odom.pose.pose.orientation
        orientation = np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        if self.Init:
            #The initial data is stored to by subtracted to all the other values as we want to start at position (0,0) and orientation 0
            self.Init = False
            self.Init_ang = orientation
            self.globalAng = self.Init_ang
            Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang),np.cos(self.Init_ang)]])
            self.Init_pos_x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y
            self.Init_pos_y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y
            self.Init_pos_z = position.z
        Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])

        #We subtract the initial values
        self.globalPos_x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y - self.Init_pos_x
        self.globalPos_y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y - self.Init_pos_y
        self.globalAng = orientation - self.Init_ang
        self.globalAng = math.atan2(np.sin(self.globalAng), np.cos(self.globalAng))

    def cropImg(self,tmask, test_hsv):
        tImage = tmask
        timgCrop = None
        center = []
        # Find only external contours
        trash, contour, hierarchy = cv2.findContours(tImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort by size
        contSort = sorted(contour, key=cv2.contourArea, reverse=True)[:1]

        # Check for Blank wall
        if hierarchy is not None:
            # Check if contour is big enough
            for n in range(len(contour)):
                M = cv2.moments(contour[n])
                if M['m00'] > 4000:
                    x, y, w, h = cv2.boundingRect(contour[n])
                    rect = cv2.minAreaRect(contour[n])
                    w = int(rect[1][0])
                    # h = int(rect[1][1])
                    if w == 0:
                        w = 120
                    x = int(M['m10'] / M['m00'])
                    y = int(M['m01'] / M['m00'])
                    y1 = int(y - h / 2)
                    y2 = int(y + h / 2)
                    x1 = int(x - w / 2)
                    x2 = int(x + w / 2)
                    # print("x1,x2,y1,y2", x1, x2, y1, y2)
                    if x1 < 0:
                        x1 = 0
                    if y1 < 0:
                        y1 = 0
                    if x2 > 360:
                        x2 = 360
                    if y2 > 240:
                        y2 = 240
                    # print("corrected x1,x2,y1,y2", x1,x2,y1,y2)

                    # Create the cropped image
                    timgCrop = test_hsv[y1:y2, x1:x2].copy()
                    # print("cropped image size", timgCrop.shape)

                    if not center:
                        center = [[x]]
                    else:
                        center.append([x])

        # If the wall is blank then AKA found no contours large enough, then just send back the original image
        if len(center) is 0:
            timgCrop = test_hsv

        return timgCrop

    def trainKNN(self):
        with open('/home/jacob/catkin_ws/src/stickney_final/src/2019imgs/train.txt', 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)
        print("Im training")
        # Read in images and blur
        kernel = np.ones((5, 5), np.uint8)
        images_bgr = [cv2.imread("/home/jacob/catkin_ws/src/stickney_final/src/2019imgs/" + lines[i][0] + ".png", 1) for i in range(len(lines))]
        blur = [cv2.GaussianBlur(images_bgr[i], (5, 5), 0) for i in range(len(images_bgr))]
        images_hsv = [cv2.cvtColor(blur[i], cv2.COLOR_BGR2HSV) for i in range(len(blur))]

        # Ranges to just get any color vs white
        lower = np.array([0, 40, 0])
        upper = np.array([180, 255, 255])

        # Generate Mask
        mask = [cv2.inRange(images_hsv[i], lower, upper) for i in range(len(images_hsv))]

        # Get rid of the arrow in the center
        imgClose = [cv2.morphologyEx(mask[i], cv2.MORPH_CLOSE, kernel) for i in range(len(mask))]

        imagesCrop = []
        for i in range(len(mask)):
            imgCrop = self.cropImg(imgClose[i], images_hsv[i])
            imagesCrop.append(imgCrop)

        # Finalize formatting to feed to KNN to train
        imagesFinal = np.array([cv2.resize(imagesCrop[i], (33, 25)) for i in range(len(imagesCrop))])
        train = imagesFinal.flatten().reshape(len(lines), 33 * 25 * 3)
        train_data = train.astype(np.float32)

        print("Just formatted all my data")
        # read in training labels
        train_labels = np.array([np.int32(lines[i][1]) for i in range(len(lines))])

        ### Train classifier
        knn = cv2.ml.KNearest_create()
        knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

        f.close()
        print("Im done training")
        return knn

    def classifyImg(self,test_img):
        """
        :param test_img: HSV image
        :return: Which direction to return
        """

        kernel = np.ones((5, 5), np.uint8)
        # Ranges to just get any color vs white
        lower = np.array([0, 40, 0])
        upper = np.array([180, 255, 255])

        msg = None
        # Generate Mask
        test_mask = cv2.inRange(test_img, lower, upper)

        # Clean up areas
        test_Close = cv2.morphologyEx(test_mask, cv2.MORPH_CLOSE, kernel)
        test_crop = self.cropImg(test_Close, test_img)

        # print("Cropped image size is: ",test_crop.shape)
        test_final = np.array(cv2.resize(test_crop, (33, 25)))
        test_final = test_final.flatten().reshape(1, 33 * 25 * 3)
        test_final = test_final.astype(np.float32)

        ret, results, neighbours, dist = self.knn.findNearest(test_final, 3)

        if ret == 0:     # Blank wall, search
            print("search")
            msg = "Search"
        elif ret == 1:   # Turn Left
            print("Left")
            msg = "Left"
        elif ret == 2:   # Turn Right
            print("Right")
            msg = "Right"
        elif ret == 3:   # Turn Around
            print("Around")
            msg = "Turn Around"
        elif ret == 4:   # Turn Around
            print("Around")
            msg = "Turn Around"
        elif ret == 5:   # Target
            print("Target")
            msg = "Target"

        return msg

    def chaseObject(self, pt):
        global pub

        msg = Twist()
        msg.linear.y = 0
        msg.linear.z = 0
        msg.angular.x = 0
        msg.angular.y = 0

        if point.x != 999:

            range = -1*(self.switch_dis-point.x)

            theta = np.deg2rad(point.y)

            # print("theta before:", theta)

            if theta > np.pi:
                theta = theta - 2*np.pi

            # print("theta after:", theta)

            if self.lastActivity != "track":
                msg.linear.x = self.linear_control.update_PID(range, dt=0.5, clear=1)
                msg.angular.z = self.angular_control.update_PID(theta, dt=0.5, clear=1)
            else:
                msg.linear.x = self.linear_control.update_PID(range, dt=0.5, clear=0)
                msg.angular.z = self.angular_control.update_PID(theta, dt=0.5, clear=0)

            # rospy.loginfo(msg)

        else:
            msg.linear.x = 0.03
            msg.linear.y = 0
            msg.linear.z = 0
            msg.angular.x = 0
            msg.angular.y = 0
            msg.angular.z = 0

        pub.publish(msg)
        # rate.sleep()

    def readImageMove(self, direction):
        """

        :param direction: string telling which way to go
        :return: This function should determine the desired heading angle based on the current odom and image class
        """
        if direction =="Left":

            # Turn to the left (90deg)
            self.desired_heading = self.globalAng + np.pi/2

            # Make sure resulting angle is between -pi and pi
            self.desired_heading = math.atan2(np.sin(self.desired_heading), np.cos(self.desired_heading))

        elif direction =="Right":
            # Turn to the left (90deg)
            self.desired_heading = self.globalAng - np.pi / 2

            # Make sure resulting angle is between -pi and pi
            self.desired_heading = math.atan2(np.sin(self.desired_heading), np.cos(self.desired_heading))

        elif direction =="Turn Around":
            # Turn to the left (90deg)
            self.desired_heading = self.globalAng + np.pi

            # Make sure resulting angle is between -pi and pi
            self.desired_heading = math.atan2(np.sin(self.desired_heading), np.cos(self.desired_heading))

        else:
            print("I do not understand the direction:", direction)

    def turnAngle(self, loc):
        """
        A function to call to turn the robot to a desired angle
        :return:
        """
        global pub
        # update Odometry
        self.update_Odometry(loc)

        # Calculate error between current and desired heading
        err_theta = self.desired_heading - self.globalAng

        # If we are heading backwards we have to adjust to try and hit pi or -pi
        if abs(err_theta) > np.pi:
            if err_theta > 0:
                err_theta -= 2 * np.pi
            else:
                err_theta += 2 * np.pi

        msg = Twist()
        msg.linear.x = 0
        msg.linear.y = 0
        msg.linear.z = 0
        msg.angular.x = 0
        msg.angular.y = 0

        if self.enum > 0:
            msg.angular.z =self.angular_control.update_PID(err_theta, dt=0.5, clear=0)
        else:
            msg.angular.z = self.angular_control.update_PID(err_theta, dt=0.5, clear=1)

        pub.publish(msg)

    def driveForward(self):
        """
        Just drive forward until you get close to a wall
        :return:
        """
        global pub

        msg = Twist()
        msg.linear.x = 0.1
        msg.linear.y = 0
        msg.linear.z = 0
        msg.angular.x = 0
        msg.angular.y = 0
        msg.angular.z = 0

        # if self.desired_heading != None:
        #     print("trying to determine heading")
        #     if self.lastActivity != "track":
        #       # Calculate error between current and desired heading
        #         err_theta = self.desired_heading - self.globalAng
        #
        #       # If we are heading backwards we have to adjust to try and hit pi or -pi
        #         if abs(err_theta) > np.pi:
        #             if err_theta > 0:
        #               err_theta -= 2 * np.pi
        #             else:
        #               err_theta += 2 * np.pi
        #
        #         msg.angular.z = self.angular_control.update_PID(err_theta, dt=0.5, clear=0)
        #
        # else:
        #     msg.angular.z = 0

        pub.publish(msg)

    def stop(self):
        global pub

        stop = Twist()
        stop.linear.x = 0
        stop.linear.y = 0
        stop.linear.z = 0
        stop.angular.x = 0
        stop.angular.y = 0
        stop.angular.z = 0
        pub.publish(stop)

    def getRange(self, rangeAhead):
        # Get the range values from the middle of the set
        middle = int(len(rangeAhead)/2)

        # Determine the range of values you want to use and get corresponding ranges
        ranges = [rangeAhead[i] for i in range(middle - 1, middle + 1)]

        # Calculate the average of those values
        avgRange = sum(ranges)/len(ranges)

        return avgRange

    def search(self, pt):
        """
        Function to just turn robot until it finds a wall
        :return:
        """
        global pub

        if point.x != 999:
            self.chaseObject(pt)
        else:
            msg = Twist()
            msg.linear.x = 0
            msg.linear.y = 0
            msg.linear.z = 0
            msg.angular.x = 0
            msg.angular.y = 0
            msg.angular.z = 0.4
            print("Im searching")
            pub.publish(msg)

    def decide(self, loc, rangeAhead, camImage):
        global rate
        global point
        # First update current location
        self.update_Odometry(loc)
        current_pose = np.array([self.globalPos_x, self.globalPos_y])

        # Range ahead of robot
        avgRange = self.getRange(rangeAhead)

        # Decide on three options
        # If you are within x meters of a wall, then use use photo and analyze for new directions
        if (self.lastActivity == "classify") and (abs(self.desired_heading - self.globalAng) >= 0.05):
            self.turnAngle(loc)

        elif avgRange <= self.switch_dis:
            #stop moving
            if self.lastActivity != "classify":
                self.stop()

            # Classify image and determine how to turn
            direction = self.classifyImg(camImage)

            # Take action
            if direction == "Target":
                rospy.sleep(20)
            elif direction == "Search":
                print("I see blank walls")
                self.search(point)

            else:
                self.readImageMove(direction)
                self.turnAngle(loc)
            self.lastActivity = "classify"

        # If you are within y-x meters of wall then follow the object
        elif avgRange > self.switch_dis and avgRange <= 1:
            if self.lastActivity != "track":
                self.stop()
            self.chaseObject(point)
            self.lastActivity = "track"

        # If you are father away than y meters from wall just drive forward
        else:
            if self.lastActivity != "forward":
                self.stop()
            self.driveForward()
            self.lastActivity = "forward"

def position(odom_data):
    global location, start
    location = odom_data
    start = True

def callback_lidar(data):
    """

    :param data: Lidar info
    :return: a range of values from a 60 degree swath in front of the robot
    """
    global frontRange
    range_max = data.range_max
    range_min = data.range_min

    # Initialize left and right side data
    left_side = []
    right_side = []

    # Get range values from for 30deg left of center and 30deg right of center
    left_side = list(data.ranges[:30])
    right_side = list(data.ranges[-30:])

    # Flip order of left data, (current order 0 -> 30deg, want to make the 30deg the new zero and 0deg the new 30deg)
    left_side.reverse()

    # Flip order of right data for same reason as left
    right_side.reverse()

    # Append range data together to get 60 degrees worth of data from with far left being 0deg and far right being 60deg
    all_ranges = left_side + right_side

    # If lidar doesn't see anything, make the range the max distance
    for i in range(len(all_ranges)):
        if all_ranges[i] < range_min:
            all_ranges[i] = range_max

    frontRange = all_ranges

def get_image(ros_data):
    """
    :param CompressedImage: data from raspicam node
    :return: a blurred HSV image from the raspicam node
    """
    global camImage

    np_arr = np.fromstring(ros_data.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Blur the image to reduce edges caused by noise or that are useless to us.
    imgBlur = cv2.GaussianBlur(image_np, (5, 5), 0)

    # Transform BGR to HSV to avoid lighting issues.
    imgHSV = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2HSV)

    camImage = imgHSV

def callback_blob(p):
    global point
    point = p


def Init():

    global pub

    # Define Publisher
    pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

    # Subscribe to odometry/obstacle information information
    rospy.Subscriber('odom', Odometry, position)

    #Subscribe to Lidar info
    rospy.Subscriber('scan', LaserScan, callback_lidar)

    # I do declare that this node hithertofore is subscribin' to the Compressed Images node.
    rospy.Subscriber("/raspicam_node/image/compressed", CompressedImage, get_image)

    # Subscribe to the follow object node
    rospy.Subscriber("thetaDist", Point, callback_blob)

    # Initialize node
    rospy.init_node('turtleboy', anonymous=True)  # make node

if __name__ == '__main__':
    try:
        Init()
    except rospy.ROSInterruptException:
        pass

    turtleBoy = Robot()

    rate = rospy.Rate(5)

    while not rospy.is_shutdown():
        # print("start is :", start)
        if start:

            # print('Im deciding')
            # Let turtleBoy decide his move
            turtleBoy.decide(location, frontRange, camImage)

        rate.sleep()
