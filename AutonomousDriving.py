# Author: Buğra Beytullahoğlu, https://github.com/beytullahoglu
# I could not be able to finish this project without help of:
# Autonomous RC Car by hamuchiwa, https://github.com/hamuchiwa/AutoRCCar
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

# Autnomous Driving code. Our neural network model will be imported and Raspberry Pi will make predictions after capturing images.
# Importing necessary packages
import cv2
import h5py
import keras
import tensorflow
import RPi.GPIO as GPIO
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
from skimage.measure import block_reduce
from keras.models import load_model

# GPIO settings
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)

# Raspberry Pi Pins
pwmPIN = 8
ControlF = 12
ControlR = 10
servoPIN = 16

# GPIO settings
GPIO.setup(pwmPIN,GPIO.OUT)
GPIO.setup(ControlF,GPIO.OUT)
GPIO.setup(ControlR,GPIO.OUT)
GPIO.setup(servoPIN, GPIO.OUT)

# PWM settings
pwm1 = GPIO.PWM(pwmPIN,100)
pwm2 = GPIO.PWM(servoPIN, 50)

# Starting PWM Duty Cycles
pwm1.start(0)
pwm2.start(5)


# Motion Functions
def servo(angle):

	angle = angle/18
	cycle = angle + 2.5
	pwm2.ChangeDutyCycle(cycle)


def forward():

	pwm1.ChangeDutyCycle(85)
	GPIO.output(ControlF,True)
	GPIO.output(ControlR,False)
	servo(50)

def forwardRight():

	pwm1.ChangeDutyCycle(90)
	GPIO.output(ControlF,True)
	GPIO.output(ControlR,False)
	servo(25)

def forwardLeft():

	pwm1.ChangeDutyCycle(90)
	GPIO.output(ControlF,True)
	GPIO.output(ControlR,False)
	servo(75)

def stop():

	pwm1.ChangeDutyCycle(0)
	GPIO.output(ControlF,True)
	GPIO.output(ControlR,False)
	servo(50)

# Function for detecting stop sign
def detect(cascade_classifier, gray_image, image):

        stop_sign = cascade_classifier.detectMultiScale(gray_image, scaleFactor=1.01, minNeighbors=5, minSize=(20, 20), maxSize(40,40))
        if (stop_sign != ()): # Checking if any stop sign is detected
            for (x, y, width, height) in cascade_obj:
                cv2.rectangle(image, (x+5, y+5), (x+width-5, y+height-5), (255, 255, 255), 2)
                cv2.putText(image, 'STOP', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return 1
        else:
            return 2

# Setting up the Pi Camera
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(320, 240))

# Small amount of sleep time to let Pi Camera be ready
time.sleep(0.1)

# Importing stop sign cascade clasiffier
stop_cascade = cv2.CascadeClassifier('stop_sign.xml') # This is a pre-trained stop sign model

#Importing neural network model
t1 = time.time()
print("Importing Neural Network Model")
model = load_model('/home/pi/Downloads/model.h5')
t2 = time.time()
print("Model has been Imported Succesfully")
print("Import time: " + str(int(t2-t1)) + " seconds")
print("Autonomous Drive is Starting in 5..")
time.sleep(1)
print("4..")
time.sleep(1)
print("3..")
time.sleep(1)
print("2..")
time.sleep(1)
print("1..")
time.sleep(1)


# Raspberry Pi will grab images and make predictions by using our neural network model 
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # Grabbing an image
    image = frame.array
    stopImage = image # Saving a separate image for stop sign detection
    gray_stopImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Cropping upper half of image
    image = image[120:240,:]
	
    # Changing BGR image to gray scale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Applying Canny Edge detection to our gray scale image
    gray_image = cv2.Canny(gray_image, 100, 200)
    
    # Showing image on a separate window
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
    if (key == ord('q')): # Autonmous Drive will be stopped if 'q' is pressed
        break
    
    # Resizing image by using block_reduce with block_size = (5,5)
    gray_image = block_reduce(gray_image, block_size = (5,5), func = np.max)
    
    # Changing pixels to either -1 or 1, which are black and white
    gray_image = (gray_image*(1/255) - 0.5)*2
    
    # Reshape image into one row
    gray_image = np.reshape(gray_image,(1,1536))
    
    # Prediting a command using neural network model
    prediction = model.predict(gray_image)
    
    # Taking index of max value of prediction array, that will be our command
    prediction = prediction.argmax(axis=1)
    
    # Calling stop sign function
    output = detect(stop_cascade, gray_stopImage, stopImage)
    
    # If any stop sign is detected, our car stops for 5 seconds
    if(output == 1):
        stop()
        time.sleep(5)
        forward()
        time.sleep(0.5)
    
    # Clearing stream to make it ready for next frame
    rawCapture.truncate(0)
    
    # Sending motion orders according to precidictions
    if(prediction == 0):
        forward()
        print("Forward")
    elif(prediction == 1):
        forwardRight()
        print("Right")
    elif(prediction == 2):
        forwardLeft()
        print("Left")
    time.sleep(0.2) #Makes 5 predictions per second
	


