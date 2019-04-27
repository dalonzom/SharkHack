from picamera import PiCamera
from time import sleep
import RPi.GPIO as GPIO
import sys
import glob 
import os
import subprocess
import numpy as np
import tensorflow as tf
#from keras.models import load_weights
import pandas as pd
from tensorflow.keras import layers

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import sys
import glob 
import os
import cv2 
from testNet import testNet
import pyttsx 

img_count=0

cam = PiCamera()
cam.rotation = -90

GPIO.setmode(GPIO.BCM)

GPIO.setup(18,GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(15,GPIO.IN, pull_up_down=GPIO.PUD_UP)

cam.start_preview()
letters = ''
while True:
    try:
        take_pic = GPIO.input(18)
        exit_state = GPIO.input(15)
        if take_pic == False:
            cam.capture('/home/pi/Desktop/images/image_%s.jpg' % img_count)
            print("button was pressed")
            sleep(0.2)
            img_count += 1
        elif exit_state == False:
            cam.stop_preview()
	    testPath = '/home/pi/Desktop/images/' 
	    for i in range(0,img_count): 
		letters = letters + (testNet([testPath, str(img_count)]))
	    print(letters) 
	    #engine = pyttsx.create_engine() 
	    #engine.say(letters)
	    #engine.runAndWait() 	
            sys.exit()
            
    except KeyboardInterrupt:
        cam.stop_preview()
        sys.exit()
        
