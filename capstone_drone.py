#!/usr/bin/env python
#################################
#                               #
#     Human tracking Drone      #
#                               #
#     Lee Do Kyu                #
#                               #
#################################
 
import time
import os
import sys
import cv2
import numpy as np
import select
import termios
import tty
try:
    import tensorflow as tf
except ImportError:
    print("unable to import TensorFlow. Is it installed?")
    print("  sudo apt install python-pip")
    print("  sudo pip install tensorflow")
    sys.exit(1)

## ROS related imports
import rospy
from std_msgs.msg import *
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from geometry_msgs.msg import Twist

## Object detection module imports
import object_detection
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

## Handgesture
import threading
import gestureCNN as myNN

minValue = 70

x0 = 400
y0 = 200
height = 200
width = 200

saveImg = False
guessGesture = True
visualize = False

lastgesture = -1

kernel = np.ones((15,15),np.uint8)
kernel2 = np.ones((1,1),np.uint8)
skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

## Which mask mode to use BinaryMask, SkinMask (True|False) OR BkgrndSubMask ('x' key)
binaryMode = True
bkgrndSubMode = False
mask = 0
bkgrnd = 0
counter = 0

## This parameter controls number of image samples to be taken PER gesture
numOfSamples = 1001
gestname = ""
path = ""
mod = 0

font = cv2.FONT_HERSHEY_SIMPLEX
size = 0.5
fx = 10
fy = 350
fh = 18
## Handgesture detection mode
mod = myNN.loadCNN(0)
## SET FRACTION OF GPU YOU WANT TO USE HERE
GPU_FRACTION = 0.4

######### Set model here ############
MODEL_NAME =  'ssd_mobilenet_v1_coco_11_06_2017'
# By default models are stored in data/models/
MODEL_PATH = os.path.join(os.path.dirname(sys.path[0]),'data','models' , MODEL_NAME)
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_PATH + '/frozen_inference_graph.pb'
######### Set the label map file here ###########
LABEL_NAME = 'mscoco_label_map.pbtxt'
# By default label maps are stored in data/labels/
PATH_TO_LABELS = os.path.join(os.path.dirname(sys.path[0]),'data','labels', LABEL_NAME)
######### Set the number of classes here #########
NUM_CLASSES = 90

##############################################################
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
##############################################################

## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`,
# we know that this corresponds to `airplane`.  Here we use internal utility functions,
# but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

## Setting the GPU options to use fraction of gpu that has been set
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = GPU_FRACTION

## Kalman Filter -------------------------------------------------
kalman = cv2.KalmanFilter(4,2)
kalman.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]],np.float32)

kalman.transitionMatrix = np.array([[1,0,1,0],
                                    [0,1,0,1],
                                    [0,0,1,0],
                                    [0,0,0,1]],np.float32)

kalman.processNoiseCov = np.array([[1,0,0,0],
                                   [0,1,0,0],
                                   [0,0,1,0],
                                   [0,0,0,1]],np.float32)

# Optical flow -----------------------------------------------------

termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
feature_params = dict( maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)
Ik_params = dict( winSize = (15,15), maxLevel=2, criteria = termination)

x_list = [-1, -1, 0, 1, 1, 1, 0, -1]
y_list = [0, 1, -1, 1, 0, -1, -1, -1]
hist_range = [i for i in range(10)]

np.random.seed(1)

global flag

flag = 0

###########################################################################
global prediction
global box
global center_x
global center_y
global kalman_center_x
global kalman_center_y
global width,height

width = 0
height = 0

box = []

measurement = np.array((2,1), np.float32)
prediction = np.zeros((2,1), np.float32)

center_x = 0
center_y = 0

center_array = np.array([])

kalman_center_x = 0	
kalman_center_y = 0

# ---------------------------------------------------------------
red = (0, 0, 255)
blue = (255,0,0)
yellow = (0,255,255)
green = (0,255,0)
thickness = 1

#----------------------------------------------------------------
global stage
global state
global count
global d
global mean
global mean_x
global mean_y
global mask_roi
global mask_blur

mask_roi = 0
mask_blur = 0

d=0
stage = 1
state = 0
count = 0
mean = 0
mean_x = 270
mean_y = 130

## Mouse event----------------------------------------------------------------------------
keep_processing = True
selection_in_progress = False

global sel
global selection_in_progress

global x
global y

sel = []
x = 0
y = 0
current_mouse_position = np.ones(2, dtype=np.int32)
#-----------------------------------------------------------------------------------------
msg = """
----------------------------
press the 't' key to takeoff

or

press the 'r' key to reset

or

press the 'l' key to landing
----------------------------
"""

keyboard={
		't':(0,0),
		'r':(0,0),
		'l':(0,0)
		}

settings = termios.tcgetattr(sys.stdin)

#########################################################################
pub=rospy.Publisher('/cmd_vel', Twist, queue_size=5)
pub_takeoff = rospy.Publisher('/ardrone/takeoff', Empty, queue_size = 1)
pub_landing = rospy.Publisher('/ardrone/land', Empty, queue_size = 1)
pub_reset = rospy.Publisher('/ardrone/reset', Empty, queue_size = 1)
#########################################################################

## Drone moving------------------------------------------------------------
def landing():

	pub_landing.publish(Empty())

def dronemove(linear_x,linear_y,linear_z,angular_z):

	rospy.on_shutdown(landing)

	twist=Twist()
	twist.linear.x = linear_x
	twist.linear.y = linear_y
	twist.linear.z = linear_z
	twist.angular.z = angular_z

	pub.publish(twist)

## Get keyboard key to 'Drone takeoff' or Drone reset'--------------------- 
def getKey():

	tty.setraw(sys.stdin.fileno())
	select.select([sys.stdin], [], [], 0)
	key = sys.stdin.read(1)
	termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

	return key

## Mouse event to start tracking --------------------------------------------
'''
def on_mouse(event, x, y, flags, params):

    global sel;
    global selection_in_progress;

    current_mouse_position[0] = x;
    current_mouse_position[1] = y;


    if event == cv2.EVENT_LBUTTONDOWN:
        sel = [];
        #print 'Start Mouse Position: '+str(x)+', '+str(y)
        sel = [x, y];
        selection_in_progress = True;

    elif event == cv2.EVENT_LBUTTONUP:
        #print 'End Mouse Position: '+str(x)+', '+str(y)
        selection_in_progress = False;
'''
## Skin Mask for Hand Gesture---------------------------------------------------------
def skinMask(frame, plot):
	global guessGesture, visualize, mod, lastgesture, saveImg
	## HSV values
	low_range = np.array([0, 50, 80])
	upper_range = np.array([30, 200, 255])
	
	cv2.rectangle(frame, (220,1),(420,201),(0,255,0),1)
	#roi = cv2.UMat(frame[y0:y0+height, x0:x0+width])
	roi = frame[1:201, 220:420]
	
	hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
	
	## Apply skin color range
	mask = cv2.inRange(hsv, low_range, upper_range)
	
	mask = cv2.erode(mask, skinkernel, iterations = 1)
	mask = cv2.dilate(mask, skinkernel, iterations = 1)
	
	## blur
	mask = cv2.GaussianBlur(mask, (15,15), 1)
	
	## bitwise and mask original frame
	res = cv2.bitwise_and(roi, roi, mask = mask)
	## color to grayscale
	res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
	
	if saveImg == True:
	    saveROIImg(res)
	elif guessGesture == True:
	    #res = cv2.UMat.get(res)
	    t = threading.Thread(target=myNN.guessGesture, args = [mod, res])
	    t.start()
	elif visualize == True:
	    layer = int(raw_input("Enter which layer to visualize "))
	    cv2.waitKey(0)
	    myNN.visualizeLayers(mod, res, layer)
	    visualize = False
	
	return res

## Kalman filter of Optical flow-----------------------------------------------------------------------
class KalmanFilter:

    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def Estimate(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted

## Detection--------------------------------------------------------------------------------------------
class Detector:

    def __init__(self):

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("image", Image, self.image_cb, queue_size=1, buff_size=2**24)
        self.image_sub = rospy.Subscriber("image", Image, self.image_mean, queue_size=1, buff_size=2**24)
        self.image_sub = rospy.Subscriber("image", Image, self.run, queue_size=1, buff_size=2**24)
        self.sess = tf.Session(graph=detection_graph)
        self.kal = KalmanFilter()
        self.prev_image=0
        self.prevTime =0
        self.prev_roi=[]
        self.roi=[]
        self.first_roi=1
        ######## Optical Flow ############
        self.track_len = 10
        self.feature_limit = 100
        self.detect_interval = 10
        self.tracks = []
        self.weight=[]
        self.tr_weight=[]
        self.frame_idx = 0
        self.blackscreen = False
        self.width = 0
        self.height = 0
        self.prev_gray =0
        self.op_hist = [0 for _ in range(256)]
        self.op_flag=1
        self.i=0
        self.roi_flag=1
        self.R=0
        self.sum_weight=0
        self.tr_sum_weight=0
        self.mov = 3
        self.tr = []

        self.sum_tr_x=320
        self.sum_tr_y=240

###########################################################################
    def get_Roi(self, image):
        #print(data)
        #print "attempt to request roi"
        #rospy.wait_for_service('Roi')
        #try:
        #s = rospy.ServiceProxy('Roi', Image2Roi)
        #roi = s(image)
        #self.roi = np.asarray(roi.Roi).reshape(-1,4)

		global state
		global width
		global height
		global center_x
		global center_y
		global flag

		if state == 3 and flag == 1:

			self.roi[1] = 200#center_x-(width/4)
			self.roi[3] = 400#center_x+(width/4)
			self.roi[0] = 100#center_y-(height/4)
			self.roi[2] = 300#center_y+(height/4)

			opt_center_x = (self.roi[1]+self.roi[3])/2
			opt_center_y = (self.roi[0]+self.roi[2])/2
			
			self.roi[0] = int(opt_center_y - 100)
			if self.roi[0] < 0:
			    self.roi[0]=0
			self.roi[1] = int(opt_center_x - 50)
			if self.roi[1] <0:
			    self.roi[1]=0
			self.roi[2] = int(opt_center_y + 100)
			if self.roi[2] > 479:
			    self.roi[2] = 479
			self.roi[3] = int(opt_center_x + 50)
			if self.roi[3] > 639:
			    self.roi[3] = 639            
		    #except rospy.ServiceException, e:
		        #print "Service call failed: %s"%e
    
    def run(self,data):

		global state
		global flag

		if state == 3 and flag == 1:

			try:
				cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
			except CvBridgeError as e:
				print(e)
			image=cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
				#image = cv_image

			if self.first_roi:

			    self.get_Roi(data)
			    self.first_roi=0

			frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

			self.width = image.shape[1]
			self.height = image.shape[0]

			curTime = time.time()

			sec = curTime - self.prevTime
	 
			self.prevTime = curTime
	 
			fps = 1.0/float(sec)

			#print "Time {0} " . format(sec) 
			#print "Estimated fps {0} " . format(fps)

			fps = "FPS : %0.1f" % fps

			
			cv2.putText(image, fps, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))

			'''
			if self.roi_flag:
			    for box in self.roi:
			        self.R = image[box[0]:box[2], box[1]: box[3]]
			    self.roi_flag = 0

			cv2.imshow("roi", self.R)
			hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	 
			hsvt = cv2.cvtColor(self.R, cv2.COLOR_BGR2HSV)

			roihist = cv2.calcHist([hsvt], [0,1], None, [180, 256], [0, 180, 0, 256])
			cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
			dst = cv2.calcBackProject([hsv],[0,1],roihist,[0,180,0,256],1)
			
			#disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
			#cv2.filter2D(dst,-1,disc,dst)
			# threshold and binary AND
			#ret,thresh = cv2.threshold(dst,50,255,0)
			#thresh = cv2.merge((thresh,thresh,thresh))
			#res = cv2.bitwise_and(image,thresh)
			#res = np.vstack((image,thresh,res))
			
			cv2.imshow("res", dst)
			'''
			self.sum_weight=0
			self.tr_sum_weight=0

			######## Optical Flow ############



			#print(np.asarray(self.tracks).reshape(-1, 1))
			#print(len(self.tracks))
			if len(self.tracks) > 0:
			        
			        self.op_hist = [0 for _ in range(10)]
			        prev_hist = [0 for _ in range(10)]
			        #self.cut_outlier(self.tracks)
			        ######## prev image, current image ###########
			        img0, img1 = self.prev_gray, frame_gray
			        ##############################################
			        
			        ######## pyramid Lucas-Kanade optical flow ##############
			        p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
			        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **Ik_params)
			        p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **Ik_params)
			        #########################################################
			        #print(p1)
			        ######## find good feature ##############
			        d = abs(p0-p0r).reshape(-1, 2).max(-1)
			        good = d < 1
			        #########################################
			        new_tracks = []
			        round_pixel=[]
			        count=0
			        #print(len(self.tracks))
			        weight=[]
			        
			        ################## Update self.track #####################
			        
			        for tr, w, (x, y), good_flag in zip(self.tracks, self.weight,  p1.reshape(-1, 2), good):
			            #i = len(new_tracks)    #index of weight 


			            is_moving = abs(tr[-1][0] - x) > self.mov or abs(tr[-1][1] - y) > self.mov
			            if not good_flag:
			                self.i+=1
			                print("not good flag " + str(self.i))
			                #_x = tr[-1][0]
			                #_y = tr[-1][1]  
			                #predict = self.kal.Estimate(_x, _y)
			                #tr.append((predict[0], predict[1]))
			                #weight.append(float(len(tr)-1)/9.0)
			                #new_tracks.append(tr)
			                #if is_moving or w > 5:
			                #if is_moving:
			                    #self.sum_weight+=len(tr)
			                    #weight.append(len(tr))
			                    #new_tracks.append(tr)                    
			                #if self.check_is_in_box(self.prev_roi, x, y):
			                continue
			            if not self.check_is_in_box(self.prev_roi, x, y):
			                print("not in box")
			                continue              
			            
			            if len(tr) > 0:
			                if  is_moving:
			                    print("tr append")
			                    tr.append((x, y))


			            if len(tr) > self.track_len:
			                del tr[0]

			            
			            #if w > np.random.uniform(0,1,1):
			            #    weight.append(float(len(tr)-1)/9.0)
			            #    new_tracks.append(tr)
			            #if w > np.random.uniform(0,1,1):
			            if is_moving:
			            #if w > 0.5:
			                
			                #if len(self.tr) < 100:                        
			                self.tr.append(tr)
			                self.tr_weight.append(len(tr))
			                
			            else:
			                new_tracks.append(tr)
			                self.sum_weight+=len(tr)
			                weight.append(len(tr))

			            if len(self.tr) > 100:
			                del self.tr[0]

			            if len(self.tr_weight) > 100:
			                del self.tr_weight[0]

			            #else:
			            #    if w > 5:
			            #        self.sum_weight+=len(tr)
			            #        weight.append(len(tr))
			            #        new_tracks.append(tr)
			            temp=[]
			            X = int(x)
			            Y = int(y)
			            #X=int(tr[-1][0])
			            #Y=int(tr[-1][1])
			            for _x, _y in zip(x_list, y_list):
			                if X+_x < 0 or Y+_y <0 or X+_x >= self.width or Y+_y >= self.height:
			                    pixel = 255
			                    #print("out")
			                else:
			                    pixel = frame_gray[Y+_y][X+_x]
			                temp.append(pixel)   
			                #print("go")
			            B2I, B = self.pixel2Binary(temp , frame_gray[Y][X])
			            one_count, u_count = self.judge_uniform(B)
			            #print(B2I)
			            #cv2.putText(image, B, (X, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue)
			            if u_count > 2:
			                self.op_hist[9]+=1
			            else:
			                self.op_hist[one_count]+=1                    

			            #self.op_hist[B2I]+=1
			            count+=1

			            round_pixel.append(temp)


			            
			        self.tracks = new_tracks
			        self.weight = weight
			        print("tracks : " + str(len(self.tracks)) + "  weights : " + str(len(self.weight)) + " tr : " + str(len(self.tr))+ " tr weights : "+ str(len(self.tr_weight)))
			        ###########################################################
	 
			        
			        if count == 0:
			            count = 1
			        self.op_hist = [float(j)/float(count) for j in self.op_hist]
			        #print("######### score ##############")
			        score = "Score :" + str(round(self.score(self.op_hist, prev_hist),3))
			        #print(score)
			        cv2.putText(image, score, (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, red)
			        #print("##############################")
			        #print(self.op_hist)

			        prev_hist = self.op_hist
			        
			        #plt.ion()
			        #plt.clf()
			        #length=[]
			        #for tr in self.tracks[0:100]:
			            #length.append(len(tr))
			        #for i in range(100-len(length)):
			            #length.append(0)

			        #plt.bar([i for i in range(len(self.weight))], self.weight)
			        #plt.xlabel('features', fontsize = 14)           
			        #plt.pause(0.01)     
			        #plt.show()
			        
			        #sleep(0.5)
			        
			        #if not self.feature_limit == len(self.tracks):
			            #self.feature_limit = len(self.tracks)
			        #	cv2.polylines(image, [np.int32(tr) for tr in self.tracks], False, blue)
			        cv2.polylines(image, [np.int32(tr) for tr in self.tracks], False, blue)

		   
		#for i in range(len(bo)):

			#if cl[i] == 1 and sc[i] > threshold:

					#box = bo[i]
					#cv2.rectangle(image,(box[1],box[0]),(box[3],box[2]),blue,2)
			                #if box[0] < box[2] and box[1] < box[3]:
			                    #image =  image[box[0]:box[2], box[1]:box[3]]
			                    #frame = CvBridge().cv2_to_imgmsg(frame, encoding="bgr8")
			        #else:
			                #image = self.prev_image
			                #print("no img")
			#print(box)
			#time.sleep(0.05)

			for box in self.roi:
			    cv2.rectangle(image,(box[1],box[0]),(box[3],box[2]),blue,2)

			# Make first feature
			if self.op_flag:
			    mask = np.zeros_like(frame_gray)
			    mask[:] = 255
			    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
			        cv2.circle(mask, (x, y), 5, 0, -1)
			    p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
			    if p is not None:
			        for x, y in np.float32(p).reshape(-1, 2):

			            if self.check_is_in_box(self.roi, x, y):
			                #self.feature_limit+=1
			                self.sum_weight+=1
			                self.tr_sum_weight+=1
			                self.tracks.append([(x,y)])
			                #self.tr.append([(x,y)])
			                self.weight.append(1)
			                #self.tr_weight.append(1)

			    if len(self.tracks) > self.feature_limit:
			        self.tracks = self.tracks[0:100]
			        self.weight = self.weight[0:100]

			    if len(self.tr) > self.feature_limit:
			        self.tr = self.tr[0:100]            
			        self.tr_weight = self.tr_weight[0:100]
			    self.op_flag=0


			
			for w in self.tr_weight:
			    self.tr_sum_weight+=w

			self.prev_roi = self.roi
			w_temp = []
			w_temp2 = []
			print(len(self.tr_weight))
			sample = self.reject_sample(self.tracks, self.weight, self.sum_weight, 10)
			sample2 = self.reject_sample(self.tr, self.tr_weight, self.tr_sum_weight, 80)

			for tr in self.tracks:
			    w_temp.append(len(tr))

			for tr in self.tr:
			    w_temp2.append(len(tr))

			self.weight = w_temp
			self.tr_weight = w_temp2
		   
			print(len(self.tr_weight))
			print("in")
			#if len(sample) > 0:

			    #plt.ion()
			    #plt.clf()
			    #plt.bar([i for i in range(len(sample))], sample)
			    #plt.xlabel('features', fontsize = 14)           
			    #plt.pause(0.01)
			print("after find optical flow : " +str(len(self.tracks))+ " "+ str(len(self.tr)))
			new_tracks=[]
			weight=[]
			for i in sample2:
			    tr = self.tr[i]
			    new_tracks.append(tr)
			    weight.append(len(tr))

			for i in sample:
			    tr = self.tracks[i]
			    #cv2.circle(image, (tr[-1][0], tr[-1][1]), 5, green, -1)
			    new_tracks.append(tr)
			    weight.append(len(tr))

			self.tracks = new_tracks
			self.weight = weight
			mask = np.zeros_like(frame_gray)
			mask[:] = 255
			for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
			    cv2.circle(mask, (x, y), 5, 0, -1)

			p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
			offset = self.feature_limit - len(self.tracks)
			
			if p is not None:
			    count_else=0
			    for i, (x, y) in enumerate(np.float32(p).reshape(-1, 2)):
			        if count_else >= offset:
			            break	
			        if self.check_is_in_box(self.roi, x, y):
			            #self.feature_limit+=1
			            count_else+=1
			            self.tracks.append([(x,y)])
			            self.weight.append(1)

			if len(self.tracks) > self.feature_limit:
			    self.tracks = self.tracks[0:100]

			    self.weight = self.weight[0:100]

		   
			if len(self.tr) > self.feature_limit:
			    self.tr = self.tr[0:100]            
			    self.tr_weight = self.tr_weight[0:100]

			for i, tr in enumerate(self.tracks):
			    cv2.circle(image, (tr[-1][0], tr[-1][1]), len(tr), green, 1)
			    cv2.putText(image, str(i) , (tr[-1][0], tr[-1][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, red)

			for tr in self.tracks:
			    self.sum_tr_x += tr[-1][0] 
			    self.sum_tr_y += tr[-1][1]

			self.sum_tr_x = int(1.0*self.sum_tr_x / len(self.tracks))
			self.sum_tr_y = int(1.0*self.sum_tr_y / len(self.tracks))
			cv2.circle(image, (self.sum_tr_x, self.sum_tr_y), 20, red, -1)
			#predict = self.kal.Estimate(self.sum_tr_x, self.sum_tr_y)

			#self.sum_tr_x = int(predict[0])
			#self.sum_tr_y = int(predict[1])

			self.roi[0] = self.sum_tr_y - 100
			if self.roi[0] < 0:
			    self.roi[0]=0
			self.roi[1] = self.sum_tr_x - 50
			if self.roi[1] <0:
			    self.roi[1]=0
			self.roi[2] = self.sum_tr_y + 100
			if self.roi[2] > 479:
			    self.roi[2] = 479
			self.roi[3] = self.sum_tr_x + 50	
			if self.roi[3] > 639:
			    self.roi[3] = 639
			#time.sleep(0.1)
			#cv2.circle(image, (sum_tr_x, sum_tr_y), 20, green, -1)
			#for tr in self.tracks:
			#    cv2.circle(image, tr[-1], 2, red, -1)

			#for tr in self.tracks:
			#    cv2.circle(image, (tr[-1][0], tr[-1][1]), 5, green, -1)

			'''
			if len(self.tracks) < self.feature_limit:
			    #self.tracks=[]
			    #print("in")
			    offset = self.feature_limit - len(self.tracks)
			    mask = np.zeros_like(frame_gray)
			    mask[:] = 255
			    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
			        cv2.circle(mask, (x, y), 5, 0, -1)
			    p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)

			    if offset <= len(sample):
			        for i in sample[0:offset]:
			            a = int(np.random.uniform(1, 5 , 1))
			            tr = self.tracks[i]
			            cv2.circle(image, (tr[-1][0], tr[-1][1]), 5, green, -1)
			            self.tracks.append(tr)
			            self.weight.append(len(tr))

			    else:
			        print(len(sample))
			        offset = offset-len(sample)
			        for i in sample:
			            tr = self.tracks[i]
			            #cv2.circle(image, (x, y), 5, green, -1)
			            cv2.circle(image, (tr[-1][0], tr[-1][1]), 5, green, -1)
			            self.tracks.append(tr)
			            self.weight.append(len(tr))                

			        if p is not None:
			            for i, (x, y) in enumerate(np.float32(p).reshape(-1, 2)):
			                if i >= offset:
			                    break	
			                if self.check_is_in_box(self.roi, x, y):
			                    #self.feature_limit+=1
			                    self.tracks.append([(x,y)])
			                    self.weight.append(1)
			else:
			    self.tracks = self.tracks[0:100]
			    self.weight = self.weight[0:100]
			'''
			print(self.weight[0:10])
			#for tr in self.tracks:
			#    cv2.circle(image, tr[-1], 5, green, -1)
			#for tr in self.tracks:
			    #print(len(tr))
			#    cv2.circle(image, tr[-1], 2, green, -1)
			#    self.make_200()
			#self.frame_idx = 1
			self.prev_gray = frame_gray
			#time.sleep(0.05)
			#for tr in self.tracks:
			#    cv2.circle(image, tr[-1], 3, blue, -1)

			print("after add new feature : " +str(len(self.tracks)) + " tr : " + str(len(self.tr)))
			print("weight : " + str(len(self.weight)) + " tr_weight : " + str(len(self.tr_weight)))
			print("############################################################")
			cv2.imshow("tracker", image)
			self.prev_img = image
			#time.sleep(0.1)
		cv2.waitKey(1)

    def reject_sample(self,tracks, weight, sum_weight, num):
        N = num
        n=0
        sample=[]
        check=0
        W = weight
        if len(W) == 0: 
            return sample
        for i in range(len(W)):
            W[i] = float(W[i])/float(sum_weight)
            check+=W[i]

        print(check)
        #print(W[0:10])
        while(n < N):
            x = np.random.uniform(0, len(tracks), 1)
            u = np.random.rand(1)
            #print(int(x))
            #print(weight[int(x)])
            if u < float(W[int(x)]):
                 sample.append(int(x))
                 n+=1
            #print("sampling : "+str(n))
        return sample
        

    def check_is_in_box(self, box, x, y):
        box = np.asarray(box)
        box = box.reshape(-1, 4)
        #print(box)
        is_it=0
        for bo in box:
            #print(bo)
            if x > bo[1] and x < bo[3] and y > bo[0] and y < bo[2]:
                is_it = 1    
                break
        return is_it

    def make_200(self):
        if len(self.tracks) > self.feature_limit:
            self.tracks = self.tracks[:self.feature_limit]
    #def cut_outlier(roi, track):

    def pixel2Binary(self,nearby, pixel):
        Binary = []
        for p in nearby:
            if p >= pixel:
                Binary.append('1')
            else:
                Binary.append('0')
        
        st = ''.join(Binary)
        #print(st)
        return int(st, 2), st

    def judge_uniform(self, binary):
        one_count=0
        u_count=0
        prev = int(binary[0])
        if prev:
            one_count+=1

        for i in range(1,8):
            curr = int(binary[i])
            if abs(prev - curr) == 1:
                u_count+=1
            if curr:
                one_count+=1
            prev = curr

        return one_count, u_count
        
    def score(self, hist1, hist2):
        score = 0
        for i, j in zip(hist1, hist2):
            M = abs(i-j)
            score = score + M*M
        return score

#######################################################################################################
    ## Getting a body color histogram
    def image_mean(self,data):   

		global roi_hist
		global state
		global mean
		global term_criteria
		global sel

		if state == 2 and mean < 50:

			cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
			image=cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
			#hist_frame = image[100:200,270:370]
			hist_frame = image[sel[1]-60:sel[1],sel[0]-30:sel[0]+30]
			hsv_roi = cv2.cvtColor(hist_frame, cv2.COLOR_BGR2HSV)
			roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
			#roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
			roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
			term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

			mean += 1

		elif state == 2 and mean >= 50:

			state = 3

    ## Main Function
    def image_cb(self,data):

		global prediction
		global box
		global center_x
		global center_y
		global kalman_center_x
		global kalman_center_y
		global stage
		global state
		global count
		global sel
		global x
		global y
		global d
		global guessGesture, visualize, mod, binaryMode, bkgrndSubMode, mask, takebkgrndSubMask, saveImg, gestname, path
		global roi_hist
		global mean
		global term_criteria
		global mean_x
		global mean_y
		global width,height
		global mask_roi
		global mask_blur
		global flag

		## If drone takeoff
		if stage == 1:

			percentage = 18

			Kp_x = 0.006
			Ki_x = 0.0002
			Kd_x = 0.000001

			Kp_y = 0.0
			Ki_y = 0.0
			Kd_y = 0.0

			Kp_z = 0.001
			Ki_z = 0.0002
			Kd_z = 0.000001

			Kp_th = 0.003
			Ki_th = 0.0003
			Kd_th = 0.000001
	
			set_time = 0.005
			current_time = time.time()
			last_time = current_time

			init_error_x = 0
			init_error_y = 0
			init_error_z = 0
			init_error_th = 0

			last_error_x = 0
			last_error_y = 0
			last_error_z = 0
			last_error_th = 0

			windup_guard = 5.0

			center_x = 0
			center_y = 0
			ratio = 0.0
			go = 0.0

			Px = 0 
			Py = 0 
			Pz = 0
			Pth = 0

			Ix = 0 
			Iy = 0 
			Iz = 0
			Ith = 0

			Dx = 0 
			Dy = 0 
			Dz = 0
			Dth = 0

			dt = 0
		
			quietMode = False
			plot = np.zeros((512,512,3), np.uint8)

			objArray = Detection2DArray()
			try:
				cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
			except CvBridgeError as e:
				print(e)
			image=cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)

			## the array based representation of the image will be used later in order to prepare the
			## result image with boxes and labels on it.
			image_np = np.asarray(image)
			## Expand dimensions since the model expects images to have shape: [1, None, None, 3]
			image_np_expanded = np.expand_dims(image_np, axis=0)
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			## Each box represents a part of the image where a particular object was detected.
			boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			## Each score represent how level of confidence for each of the objects.
			## Score is shown on the result image, together with the class label.
			scores = detection_graph.get_tensor_by_name('detection_scores:0')
			classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')

			start_time = time.time()

			(boxes, scores, classes, num_detections) = self.sess.run([boxes, scores, classes, num_detections],
				feed_dict={image_tensor: image_np_expanded})

			end_time = time.time()

			#print("Elapsed Time:", end_time-start_time)

			im_height, im_width,_ = image.shape
			boxes_list = [None for i in range(boxes.shape[1])]
			for i in range(boxes.shape[1]):
				boxes_list[i] = (int(boxes[0,i,0] * im_height),
				            int(boxes[0,i,1]*im_width),
				            int(boxes[0,i,2] * im_height),
				            int(boxes[0,i,3]*im_width))

			bo = boxes_list
			sc = scores[0].tolist()
			cl = [int(x) for x in classes[0].tolist()]
			nu = int(num_detections[0])

			threshold = 0.7

			cv2.namedWindow('tracker')
			#cv2.setMouseCallback('tracker', on_mouse, 0);

			## Hand gesture wait
			if state == 0:
			
				print('Show me your Hand!')
				dronemove(0,0,0,0)

				frame = cv2.flip(image, 3)
				#frame = cv2.resize(frame, (640,360))
				roi = skinMask(frame, plot)
				#roi2 = binaryMask(frame, plot)
	
				plot = np.zeros((512,512,3), np.uint8)
				plot = myNN.update(plot)
				state = myNN.state(state)

				cv2.imshow("Gesture", frame)
				cv2.imshow('Gesture Probability',plot)
				cv2.imshow('ROI', roi)

			## Tracking hand gesture detected
			if state == 1:

				print('Start tracking')
				x = 320
				y = 180
				sel = [x, y]
				state = 2

			
			#if state == 3:
				
				#image_roi = image[100:200,300:400]
				#hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
				#mask = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
				#mask = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)
				#mask_roi = mask[100:200,300:400]
				#_, track_window = cv2.meanShift(mask, (mean_x, mean_y, 100, 100), term_criteria)
				#mean_x, mean_y, w, h = track_window
			
				#cv2.rectangle(image, (mean_x, mean_y), (mean_x + w, mean_y + h), red, 2)

				#sel = [(mean_x + mean_x + w)/2, (mean_y + mean_y + h)/2]

			for i in range(len(bo)):

				## Human detection
				if cl[i] == 1 and sc[i] > threshold:

					box = bo[i]
					cv2.rectangle(image,(box[1],box[0]),(box[3],box[2]),blue,2)

					if (len(sel) > 1) and sel[0] > box[1] and sel[0] < box[3] and sel[1] > box[0] and sel[1] < box[2]:

						if state == 2:

							center_x = int((box[1] + box[3])/2)
							center_y = int((box[0] + box[2])/2)

							sel[0] = center_x
							sel[1] = center_y
	
							cv2.rectangle(image,(sel[0]-30,sel[1]-60),(sel[0]+30,sel[1]),red,2)
							cv2.circle(image, (sel[0],sel[1]-30), 10,red,2)

						elif state == 3:

							center_x = int((box[1] + box[3])/2)
							center_y = int((box[0] + box[2])/2)

							width = box[3] - box[1]
							height = box[2] - box[0]

							## Tracking with body color histogram

							#image_roi = image[100:200,300:400]
							hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
							#mask = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
							mask = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)
							mask_blur = cv2.medianBlur(mask,3)
							mask_roi = mask_blur[center_y-(height/4):center_y+(height/4),center_x-(width/4):center_x+(width/4)]
							_, track_window = cv2.meanShift(mask, (mean_x, mean_y, 100, 100), term_criteria)
							mean_x, mean_y, w, h = track_window
			
							#cv2.rectangle(image, (mean_x, mean_y), (mean_x + w, mean_y + h), red, 2)

							sel = [(mean_x + mean_x + w)/2, (mean_y + mean_y + h)/2]

							flag = 1

						cv2.circle(image, (center_x,center_y-30), 10, blue,2)

						center_array = np.array([center_x,center_y],np.float32)
						#print(center_array)
						kalman.correct(center_array)
						prediction = kalman.predict()

						ratio = float(((box[3]-box[1])*(box[2]-box[0]))*100/(640*360))

						go = float(percentage - ratio)

						if go < -percentage:

							go = -percentage

						str = ("g : %.2f " % ratio)

						#cv2.putText(image, str, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, blue)
						#cv2.cv.PutText(cv2.cv.fromarray(image), str(i + 1), (x, y + h), font, (0, 255, 255))

						cv2.rectangle(image, (prediction[0] - (0.5 * width), prediction[1] - (0.5 * height)),(prediction[0] + (0.5 * width), prediction[1] + (0.5 * height)), green, 2)

						kalman_center_x = int(((prediction[0] - (0.2 * box[3])) + (prediction[0] + (0.2 * box[3])))/2)
						kalman_center_y = int(((prediction[1] - (0.2 * box[2]) - 30) + (prediction[1] + (0.2 * box[2]) - 30))/2)

						cv2.circle(image, (kalman_center_x,kalman_center_y), 10, green,2)
						#cv2.line(image,(320,180),(center_x,180),red,2)
						#cv2.line(image,(center_x,180),(center_x,center_y),red,2)

						error_x = go
						error_y = (320 - kalman_center_x)
						error_z = (180 - kalman_center_y)
						error_th = (320 - kalman_center_x)

						if error_x > -8 and error_x < 8 and error_y > -15 and error_y < 15 and error_z > -15 and error_z < 15 and error_th > -5 and error_th < 5:

							print('Target Lock On')
							dronemove(0,0,0,0)
		
						else:
						
							current_time = time.time()
							dt = current_time - last_time

							de_x = error_x - last_error_x
							de_y = error_y - last_error_y
							de_z = error_z - last_error_z
							de_th = error_th - last_error_th

							if (dt >= set_time):

								Px = Kp_x * error_x
								Py = Kp_y * error_y
								Pz = Kp_z * error_z
								Pth = Kp_th * error_th

								Ix += error_x * dt
								Iy += error_y * dt
								Iz += error_z * dt
								Ith += error_th * dt

								if (Ix < -windup_guard):
									Ix = -windup_guard
								elif (Iy < -windup_guard):
									Iy = -windup_guard
								elif (Iz < -windup_guard):
									Iz = -windup_guard
								elif (Ith < -windup_guard):
									Ith = -windup_guard

								elif (Ix > windup_guard):
									Ix = windup_guard
								elif (Iy > windup_guard):
									Iy = windup_guard
								elif (Iz > windup_guard):
									Iz = windup_guard
								elif (Ith > windup_guard):
									Ith = windup_guard

								Dx = 0
								Dy = 0
								Dz = 0
								Dth = 0

								if dt > 0:

									Dx = de_x / dt
									Dy = de_y / dt
									Dz = de_z / dt
									Dth = de_th / dt

								last_time = current_time
								last_error_x = error_x
								last_error_y = error_y
								last_error_z = error_z
								last_error_th = error_th

								a = Px + (Ki_x * Ix) + (Kd_x * Dx)
								b = Py + (Ki_y * Iy) + (Kd_y * Dy)
								c = Pz + (Ki_z * Iz) + (Kd_z * Dz)
								d = Pth + (Ki_th * Ith) + (Kd_th * Dth)

								count += 1

								if count > 10:
									#print('vel =', a, b, c, d)
									print('target tracking')
									dronemove(a, b, c, d)
				
					elif (len(sel) > 1) and sel[0] < box[1] and sel[0] > box[3] and sel[1] < box[0] and sel[1] > box[2]:
		
						print('find the target')
						#print(d)
						dronemove(-0.01,0,0,d/8)
						cv2.rectangle(image, (prediction[0] - (0.5 * width), prediction[1] - (0.5 * height)),(prediction[0] + (0.5 * width), prediction[1] + (0.5 * height)), green, 2)
						kalman_center_x = int(((prediction[0] - (0.2 * box[3])) + (prediction[0] + (0.2 * box[3])))/2)
						kalman_center_y = int(((prediction[1] - (0.2 * box[2]) - 30) + (prediction[1] + (0.2 * box[2]) - 30))/2)
						cv2.circle(image, (kalman_center_x,kalman_center_y), 10, green,2)
			
				elif sc[0] <= threshold and (len(sel) > 1):

					print('miss the target')
					#print(d)
					dronemove(-0.01,0,0,d/8)
					cv2.rectangle(image, (prediction[0] - (0.5 * width), prediction[1] - (0.5 * height)),(prediction[0] + (0.5 * width), prediction[1] + (0.5 * height)), green, 2)
					kalman_center_x = int(((prediction[0] - (0.2 * box[3])) + (prediction[0] + (0.2 * box[3])))/2)
					kalman_center_y = int(((prediction[1] - (0.2 * box[2]) - 30) + (prediction[1] + (0.2 * box[2]) - 30))/2)

					cv2.circle(image, (kalman_center_x,kalman_center_y), 10, green,2)

			if (len(sel) > 1):

				cv2.rectangle(image, (prediction[0] - (0.5 * width), prediction[1] - (0.5 * height)),(prediction[0] + (0.5 * width), prediction[1] + (0.5 * height)), green, 2)
				kalman_center_x = int(((prediction[0] - (0.2 * box[3])) + (prediction[0] + (0.2 * box[3])))/2)
				kalman_center_y = int(((prediction[1] - (0.2 * box[2]) - 30) + (prediction[1] + (0.2 * box[2]) - 30))/2)

				cv2.circle(image, (kalman_center_x,kalman_center_y), 10, green,2)
				cv2.circle(image, (sel[0],sel[1]), 10,red,2)

			cv2.imshow("tracker", image)
			cv2.imshow("Mask", mask_blur)
			cv2.imshow("mask_roi",mask_roi)
			cv2.waitKey(1)

		else:

			stage = 0
			takeoff()


def takeoff():

	global stage

	if stage == 0:

		print msg
		key = getKey()

		if key in keyboard.keys():

			if (key == 't'):

				print('Take off')
				stage = 1
				pub_takeoff.publish(Empty());
				
			elif (key == 'r'):
				
				print('reset')
				pub_reset.publish(Empty());

			elif (key == 'l'):

				print('land')
				pub_landing.publish(Empty())

	else:

		stage = 0


def main(args):
    rospy.init_node('detector_node')
    obj=Detector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("ShutDown")
    cv2.destroyAllWindows()

if __name__=='__main__':
	main(sys.argv)
