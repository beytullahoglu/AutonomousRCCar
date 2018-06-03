# Author: Buğra Beytullahoğlu, https://github.com/beytullahoglu
# I could not be able to finish this project without help of:
# Autonomous RC Car by hamuchiwa, https://github.com/hamuchiwa/AutoRCCar
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

# This Code will be used to collect user input with images. All data collected will be used in neural network to train our car.
# Importing necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import RPi.GPIO as GPIO
import numpy as np
from skimage.measure import block_reduce

# GPIO setup
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)

# DC and Servo Motor pins
pwmPIN = 8
ControlF = 12
ControlR = 10
servoPIN = 16

# Setting all pins above to be output pins.
GPIO.setup(pwmPIN,GPIO.OUT)
GPIO.setup(ControlF,GPIO.OUT)
GPIO.setup(ControlR,GPIO.OUT)
GPIO.setup(servoPIN, GPIO.OUT)

# Defining PWMs and their frequencies
pwm1 = GPIO.PWM(pwmPIN,100)
pwm2 = GPIO.PWM(servoPIN, 50)

# Starting PWM DutyCycles
pwm1.start(0)
pwm2.start(5) # Servo motor set to look forward at start. You most likely need to change it according to servo motor you use.


# Motion functions: Servo, Forward, ForwardRight, ForwardLeft and Stop, respectively.
# DC Motor PWM cycles should be arranged according to road conditions. These numbers works well with our DC Motor providing a flat road.
# Servo Motor angles should be set according to your servo motor. These numbers works well with our servo
def servo(angle):

	angle = angle/18  
	cycle = angle + 2.5  # We have a servo motor works between 2.5% and 12.5% duty cycles. 
	pwm2.ChangeDutyCycle(cycle)

def forward():

	pwm1.ChangeDutyCycle(70)
	GPIO.output(ControlF,True)
	GPIO.output(ControlR,False)
	servo(50)

def forwardRight():

	pwm1.ChangeDutyCycle(65)
	GPIO.output(ControlF,True)
	GPIO.output(ControlR,False)
	servo(25)

def forwardLeft():

	pwm1.ChangeDutyCycle(65)
	GPIO.output(ControlF,True)
	GPIO.output(ControlR,False)
	servo(75)

def stop():

	pwm1.ChangeDutyCycle(0)
	GPIO.output(ControlF,True)
	GPIO.output(ControlR,False)
	servo(50)

#Creating two numpy arrays. One of which is for images, and other one is for commands. 
image_array = np.zeros((1,1536))
command_array = np.zeros((1,4))

#Command arrays. We will stack one of these arrays to command_array according to user command.
fw = [1,0,0,0] # Forward
st = [0,1,0,0] # Stop
fwr = [0,0,1,0] # Forward Right
fwl = [0,0,0,1] # Forward Left

# Initializing Pi Camera and frame features such as resolution and frame rate.
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(320, 240))

# Sleep a small amount of time to let Pi Camera to warm-up
time.sleep(0.1)

# Here, user starts driving the car and Raspberry Pi constantly saves one image and one command every time user pushes keyboard
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    
    # Grabbing an image from video, this image is grabbed as a numpy array
    image = frame.array
    
    # Changing BGR image to Gray Scale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Cropping upper half of the image, because upper half of the image does not contain road lines
    image = image[120:240,:]
    
    # Applying Canny Edge Detection
    image = cv2.Canny(image, 100, 200)
    
    # Showing frames on a separate window
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
    
    # Resizing canny image with a block size of (5,5). This function scanning (5,5) arrays and decrease to one pixel which is maximum of (5,5) array. 
    image = block_reduce(image, block_size = (5,5), func = np.max)
    image = np.reshape(image,(1,1536))
    
    # Clearing the stream to make it ready for next frame
    rawCapture.truncate(0)

	# Keyboard commands. Raspberry Pi constantly takes input and saves the corresponding image 
	if key == ord("w"):
		forward()
		image_array = np.vstack((image_array, image))
		command_array = np.vstack((command_array, fw))
	elif key == ord("d"):
		forwardRight()
		image_array = np.vstack((image_array, image))
		command_array = np.vstack((command_array, fwr))
	elif key == ord("a"):
		forwardLeft()
		image_array = np.vstack((image_array, image))
		command_array = np.vstack((command_array, fwl))
	elif key == ord("s"):
		stop()
		image_array = np.vstack((image_array, image))
		command_array = np.vstack((command_array, st))
	if key == ord("q"): # Driving will be stopped if user pushes 'q'
		break

#Save arrays to different csv files, these files will be used in neural network training
np.savetxt('/home/pi/training_images/imgdata.csv',image_array, delimiter = ",")
np.savetxt("/home/pi/training_images/commanddata.csv",command_array, delimiter = ",")

#Print number of images 
print image_array.shape
print command_array.shape

