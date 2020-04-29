#!/usr/bin/env python
"""
For lab4 in AE-7785 Intro to Robotics

Author: Jacob Stickney
"""
import rospy
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist
from tf.msg import *
import math
import numpy as np

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

    def __init__(self, goals):
        global pub
        self.Init = True
        # Initialize the PID controllers
        self.linear_control = PID(0.3, 0.15, 0.1, -0.1)
        self.angular_control = PID(0.9, 0.2, 0.1, -0.1)
        self.goals = goals
        self.enum = 0
        self.Init_ang = 0
        self.Init_pos_x = 0
        self.Init_pos_y = 0
        self.Init_pos_z = 0
        self.globalPos_x = 0
        self.globalPos_y = 0
        self.globalAng = 0
        self.lastActivity = None
        self.switch_loc = None
        self.avoid_angle = None

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

    def decide(self,loc, obs_x, obs_y):

        # First update current location
        self.update_Odometry(loc)
        current_pose = np.array([self.globalPos_x, self.globalPos_y])

        # Decide on three options
        # If you see an obstacle, avoid it
        if obs_x != 999:
            self.avoid(obs_x, obs_y)
            if self.lastActivity != 'Avoid':
                self.switch_loc = np.array((self.globalPos_x, self.globalPos_y))
            self.lastActivity = 'Avoid'

        # If you don't see an obstacle any more but were previously avoiding one keep avoid a little while longer
        elif (obs_x == 999) and (self.lastActivity == "Avoid") and \
                (np.linalg.norm(self.switch_loc - current_pose) < 0.3):
            self.avoid(obs_x,obs_y)
            self.lastActivity = 'Avoid'

        else:
            self.goGoal()
            self.lastActivity = 'Goal'

    def avoid(self, obs_x, obs_y):
        global pub

        if obs_x == 999: # should only execute if you were previously avoiding and just haven't moved far enough to stop avoiding
            # Angular error
            err_theta = self.avoid_angle - self.globalAng
            err_theta = math.atan2(np.sin(err_theta),np.cos(err_theta))

            # Should already have a calculated avoidance angle, just keep controlling
            msg = Twist()
            msg.linear.x = 0.1
            msg.linear.y = 0
            msg.linear.z = 0
            msg.angular.x = 0
            msg.angular.y = 0
            msg.angular.z = self.angular_control.update_PID(err_theta, dt=0.5, clear=0)

            # rospy.loginfo(msg)
            print("I'm avoiding an obstacle")
            pub.publish(msg)

        else:

            # Determine angle of obstacle
            obs_y -= 45  # Set 0deg to heading of robot
            obs_y *= -1  # Turning left is positive, right is negative
            obs_rad = np.deg2rad(obs_y)  # Convert to radians

            # Determine desired angle, parallel to the wall
            if obs_y <= 0:
                self.avoid_angle = self.globalAng + obs_rad + np.pi/2
            else:
                self.avoid_angle = self.globalAng + obs_rad - np.pi/2

            self.avoid_angle = math.atan2(np.sin(self.avoid_angle),np.cos(self.avoid_angle))
            # Determine error between desired angle and current angle
            err_theta = self.avoid_angle - self.globalAng
            err_theta = math.atan2(np.sin(err_theta), np.cos(err_theta))

            #Send control
            msg = Twist()
            msg.linear.x = 0.2
            msg.linear.y = 0
            msg.linear.z = 0
            msg.angular.x = 0
            msg.angular.y = 0
            msg.angular.z = self.angular_control.update_PID(err_theta, dt=0.5, clear=1)

            # Send message
            # rospy.loginfo(msg)
            print("I'm still avoiding an obstacle")
            pub.publish(msg)

    def goGoal(self):
        global pub

        cur_loc = np.array((self.globalPos_x, self.globalPos_y))

        # What is current goal
        goal_x = self.goals[self.enum][0]
        goal_y = self.goals[self.enum][1]
        goal = np.array((goal_x, goal_y))

        # Is current location at a goal?
        dist = np.linalg.norm(cur_loc - goal)
        if dist < 0.05:  # if close enough then pause and set next goal

            #Stop moving
            stop = Twist()
            stop.linear.x = 0
            stop.linear.y = 0
            stop.linear.z = 0
            stop.angular.x = 0
            stop.angular.y = 0
            stop.angular.z = 0
            pub.publish(stop)

            # Set the next goal point
            self.enum += 1
            print("I made it to goal:", self.enum)

            # Pause
            rospy.sleep(2)

        else:
            # Determine desired heading and heading error
            x1 = goal_y - self.globalPos_y
            x2 = goal_x - self.globalPos_x
            theta_des = np.arctan2(x1, x2)
            err_theta = theta_des - self.globalAng

            # If we are heading backwards we have to adjust to try and hit pi or -pi
            if abs(err_theta) > np.pi:
                if err_theta > 0:
                    err_theta -= 2*np.pi
                else:
                    err_theta += 2*np.pi

            # Initialize msg
            msg = Twist()
            msg.linear.y = 0
            msg.linear.z = 0
            msg.angular.x = 0
            msg.angular.y = 0

            # Update controller values
            if self.lastActivity == 'Avoid':
                msg.linear.x = self.linear_control.update_PID(dist, dt=0.5, clear=1)
                msg.angular.z = self.angular_control.update_PID(err_theta, dt=0.5, clear=1)
            else:
                msg.linear.x = self.linear_control.update_PID(dist, dt=0.5, clear=0)
                msg.angular.z = self.angular_control.update_PID(err_theta, dt=0.5, clear=0)

            # rospy.loginfo(msg)
            pub.publish(msg)


def position(odom_data):
    global location, start
    location = odom_data
    start = True
def obs(data):
    global x_pos
    global y_pos
    x_pos = data.x
    y_pos = data.y

def Init():

    global pub

    # Define Publisher
    pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)


    # Subscribe to odometry/obstacle information information
    rospy.Subscriber('odom', Odometry, position)
    rospy.Subscriber('obstacle', Point, obs)

    # Initialize node
    rospy.init_node('odometry', anonymous=True)  # make node

if __name__ == '__main__':
    try:
        Init()
    except rospy.ROSInterruptException:
        pass

    # Read in text file and set up list of goal points
    text_file = open("/home/jacob/catkin_ws/src/stickney_go_to_goal/src/wayPoints.txt",'r')
    goals=[]
    for line in text_file:
        goals.append(line.strip().split('\n'))

    text_file.close()

    for i,s in enumerate(goals):
        goals[i][0] = s[0].split()
        goals[i] = [val for sublist in goals[i] for val in sublist]

    for i,s in enumerate(goals):
        goals[i] = [float(s[0]),float(s[1])]

    # Initialize Robot with his goal positions
    turtleBoy = Robot(goals)

    rate = rospy.Rate(5)

    while not rospy.is_shutdown():

        # if x_pos == 999:  # Go to da Goal!! Go turtleBoy go!
        #     turtleBoy.goGoal(location)
        # else:   # Oh no an obstacle, lookout turtleBoy!
        #     turtleBoy.avoid(location, x_pos, y_pos)

        if start:
            # Let turtleBoy decide his move
            turtleBoy.decide(location, x_pos, y_pos)

        rate.sleep()