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


path = '/home/marissa/Downloads/aslalphabet/asl_alphabet_test_real_world_+_kaggle/A/A0001_test.jpg'
temp = testNet(path)
print(temp)
#temp = subprocess.Popen(["python",  "/home/marissa/SharkHack/testNet.py", "/home/marissa/Downloads/aslalphabet/asl_alphabet_test_real_world_+_kaggle/A/A0001_test.jpg"], stdout=subprocess.PIPE, 
#            stderr=subprocess.STDOUT)

#temp = subprocess.call(["python",  "/home/marissa/SharkHack/testNet.py", "/home/marissa/Downloads/aslalphabet/asl_alphabet_test_real_world_+_kaggle/A/A0001_test.jpg"], shell = True)
#print(temp)
#stdout,stderr = temp.communicate()
#print(stdout)
#print(stderr)
