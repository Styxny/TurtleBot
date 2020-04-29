#!/usr/bin/env python
# Detect and object with the camera. 

import rospy
from sensor_msgs.msg import CompressedImage
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from std_msgs.msg import String
import numpy as np


###################################
## VARIABLE DECLARATION AND SETUP
###################################

bridge = CvBridge() #Bridge converts the image from ros to openCV

name = "Object!"

lower= np.array([0, 40, 44],np.uint8) # Array (H,S,V) for the lower threshold bound of the HSV image
upper= np.array([180, 230, 255],np.uint8) # Array (H,S,V) for the upper threshold bound of the HSV image
error = np.array([0,0,0],np.uint8) # Array of error widths to create the upper and lower threshold bounds above.


titleTracker = "Color Tracker"      # Debugging Image Title
titleOriginal = "Original Image"    # Debugging Image Title
titleMask = "Mask Image"            # Debugging Image Title
debug = False                    # True - shows the images. False - Does not show the images.

width = 360                         # Width of the image, this is sent in our point message as the z-component to know the zero point in the frame.
blurSize = 9                        # Blur Kernel Size
morphOpSize = 5                     # Closing and Opening Kernel Size


maxObjects = 1                      # Max number of object to detect.
minObjectArea = 1800                 # Min number of pixels for an object to be recognized.

start = False                       # Set to true when first image is acquired and will start the program.

update = False                      # True - When a new point has been found and can be published. False - Otherwise.

mose   = False


###################################
## Function Declaration
###################################

def mouseEvent(event, x, y, flags, param):
    # The mouse event is connected to the "Original Image Window" and triggers the event when the user click on the image.
    # The HSV values of the pixel that was clicked are used to determine the HSV lower and upper bounds.
    global imgHSV
    global lower
    global upper
    global error
    global mose
    
    if event == cv2.EVENT_LBUTTONDOWN:
        lower = imgHSV[y,x,:]
        upper = imgHSV[y,x,:]     

        lower = cv2.subtract(lower,error)
        upper = cv2.add(upper,error)
        mose = True


        #rospy.loginfo("Hue Range: [%d  %d]",lower[0], upper[0])
        #rospy.loginfo("Sat Range: [%d  %d]",lower[1], upper[1])
        #rospy.loginfo("Value Range: [%d  %d]",lower[2], upper[2])

def morphOps(binaryMatrix, kernelSize):
    # Morphological operations (open and close) used to reduce noise in the acquired image.
    kernel = np.ones((kernelSize,kernelSize), np.uint8)
    tempFix = cv2.morphologyEx(binaryMatrix,cv2.MORPH_CLOSE, kernel)   # Fill in holes
    fix = cv2.morphologyEx(tempFix,cv2.MORPH_OPEN, kernel)             # Get rid of noise
    return fix

def drawCOM(frame, x, y, name):
    cv2.circle(frame,(x,y),5,(0,255,0))
    cv2.putText(frame,name,(x-30,y-25),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)


def findObjects(binaryMatrix):
    #Finds the location of the desired object in the image.
    output = []
    trash, contours, hierarchy = cv2.findContours(binaryMatrix, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Contours the image to find blobs of the same color   
    cont = sorted(contours, key = cv2.contourArea, reverse = True)[:maxObjects]                   # Sorts the blobs by size (Largest to smallest) 

    # Find the center of mass of the blob if there are any
    if len(cont) > 0:
        for i in range (0,len(cont)):
            M = cv2.moments(cont[i])
            if M['m00'] > minObjectArea:                                   # Check if the total area of the contour is large enough to care about!
                rect = cv2.minAreaRect(cont[0])
                w = int(rect[1][0])
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])
                if(debug):
                    cv2.drawContours(imgTrack, cont[i], -1, (255,0,0), 3) # Draws the contour.
                    drawCOM(imgTrack,x,y,name)
                if output == []:
                    output = [[x,w]]
                else:
                    output.append[[x,w]]
    return output


def get_image(CompressedImage):
    # get_image is the main function to find the circles in the image. Get_image triggers each time a new image arrives.

    # All the images used to find the ball are made global so we can display them durring debugging.
    global imgBGR
    global imgHSV
    global imgBLUR
    global mask
    global imgMorphOps
    global imgTrack

    # Needed parameters from outside this function (lazy and globaling them).
    global p
    global update
    global start
    global morphOpSize
    global blurSize
    global width
    global pt

    # The "CompressedImage" is transformed to a color image in BGR space and is store in "imgBGR"
    imgBGR = bridge.compressed_imgmsg_to_cv2(CompressedImage, "bgr8")

    # height and width of the image to pass along to the PID controller as the reference point.
    height, width = imgBGR.shape[:2]

    # Image used to draw things on!
    imgTrack = imgBGR.copy()
    
    # Blur the image to reduce edges caused by noise or that are useless to us.
    imgBlur = cv2.GaussianBlur(imgBGR,(blurSize,blurSize),0)

    # Transform BGR to HSV to avoid lighting issues.
    imgHSV = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2HSV)	
    
    # Threshold the image using the selected lower and upper bounds of the color of the object.
    mask = cv2.inRange(imgHSV, lower, upper)

    # To get rid of noise and fill in gaps in our object use open and close.
    imgMorphOps = morphOps(mask, morphOpSize)

    centers = findObjects(imgMorphOps)

    #print ("centers", not centers)


    # Not always, the houghCircles function finds circle, so a None inspection is made
    if not centers: 
        #print("hello")
        #If no object was found, sends bogus numbers.
        pt = Point()

        pt.x = 999
        pt.y = 999
        pt.z = 999                
        update = True

    elif centers is not []:
        for i in centers:
            # The x position of the center of the object, the width of the object, and the width of the image.
            p = Point(i[0],i[1],width)

            pt = Point()

            pt.x = p.x
            pt.y=p.y
            pt.z=p.z
            # Bool to indicate the need to publish new information
            update = True
        

    # Once the first image has been processed set start to True to display.
    start = True




def Init():

    # Creates the node, the publisher, and subscribes to the compressedImage.

    global pub
    pub = rospy.Publisher('imageLocation', Point, queue_size=1)
    
    #I declare that  the find_ball is subcribing to the Compressed Images node.
    rospy.Subscriber("/raspicam_node/image/compressed",CompressedImage, get_image)
    
    #Initializate the node and gives a name, in this case, 'find_ball'
    rospy.init_node('find_object', anonymous=True)

    #Create a publisher that will be publishing Geometric message Points
    


###################################
## MAIN
###################################

if __name__ == '__main__':
    try:
        Init()
    except rospy.ROSInterruptException:
        pass


#Rate is used for manage the looping desired rate by using the method 'sleep'
rate = rospy.Rate(10)

# Create Debugging Windows
if(debug):
    cv2.namedWindow(titleTracker, cv2.WINDOW_AUTOSIZE )
    cv2.moveWindow(titleTracker, 620, 50)
    cv2.namedWindow(titleMask, cv2.WINDOW_AUTOSIZE )
    cv2.moveWindow(titleMask, 1240, 50)
    cv2.namedWindow(titleOriginal, cv2.WINDOW_AUTOSIZE )
    cv2.moveWindow(titleOriginal, 50, 50)

# The mousecallback is connected to the "Original Image window" for the user to select the corresponding color
# cv2.setMouseCallback(titleOriginal,mouseEvent)

while not rospy.is_shutdown():
    # This is the infinite loop that keep the program running
    
    # If the first image arrived, the start = True
    if start:

        # Display the image
        if debug:
            cv2.imshow(titleOriginal,imgBGR)
            cv2.imshow(titleMask,mask)
            cv2.imshow(titleTracker, imgTrack)

        # If a new point was found, then update is True and the point is publish
        if update:   

            #print ("point",pt)
            # rospy.loginfo(pt)
            pub.publish(pt)
            update = False

        rate.sleep()

        # User's options to interact with the software
        # k = cv2.waitKey(10)
        #
        # if k == 49: #number 1
        #     #Decrease the Hue error
        #     error[0] = error[0] - 1
        #     if (error[0] < 0):
        #         error[0] = 0
        #     k = 0
        #     rospy.loginfo("Color Error: %d",error[0])
        # elif k == 50: #number 2
        #     #Increase the Hue error
        #     error[0] = error[0] + 1
        #     if (error[0] > 50):
        #         error[0] = 50
        #     k = 0
        #     rospy.loginfo("Color Error: %d",error[0])
        # elif k == 51: #numer 3
        #     #Decrease the morphOp kernel size
        #     morphOpSize = morphOpSize - 2
        #     if morphOpSize < 1:
        #         morphOpSize = 1
        #     k = 0
        #     rospy.loginfo("Kernel size for close and open: %d",morphOpSize)
        #
        # elif k == 52: #number 4
        #     #Increase the morphOp kernel size
        #     morphOpSize = morphOpSize + 2
        #     k = 0
        #     rospy.loginfo("Kernel size for close and open: %d",morphOpSize)
        # elif k == 53: #numer 5
        #     #Decrease the blur size
        #     blurSize = blurSize - 2
        #     if blurSize < 1:
        #         blurSize = 1
        #     k = 0
        #     rospy.loginfo("Bluring kernel size: %d",blurSize)
        #
        # elif k == 54: #number 6
        #     #Increase the blur size
        #     blurSize = blurSize + 2
        #     k = 0
        #     rospy.loginfo("Bluring kernel size: %d",blurSize)
        #
        # elif k == 55: #number 7
        #     #Decrease the min pixel area of the tracked object
        #     minObjectArea = minObjectArea - 1
        #     if minObjectArea < 1:
        #         minObjectArea = 1
        #     k = 0
        #     rospy.loginfo("Min object pixel area: %d",blurSize)
        #
        # elif k == 56: #number 8
        #     #Increase the min pixel area of the tracked object
        #     minObjectArea = minObjectArea + 1
        #     k = 0
        #     rospy.loginfo("Min object pixel area: %d",blurSize)

