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


def testNet(path): 
	testPath = '/home/marissa/Downloads/aslalphabet/asl_alphabet_test_real_world_+_kaggle'
	numTest = 0 
	for r, d, f in os.walk(testPath):
		for file in f:
			numTest += 1

	trainPath = '/home/marissa/Downloads/aslalphabet/asl_alphabet_train_kaggle'
	numTrain = 0 
	for r, d, f in os.walk(trainPath):
		for file in f:
			numTrain += 1

	print(numTest)
	print(numTrain)
	IMG_SIZE = 200
	NB_CHANNELS = 3 
	BATCH_SIZE = 64 
	NB_TRAIN_IMG = numTrain
	NB_VALID_IMG = numTest

	cnn = Sequential()

	cnn.add(Conv2D(filters=32, 
		       kernel_size=(2,2), 
		       strides=(1,1),
		       padding='same',
		       input_shape=(IMG_SIZE,IMG_SIZE,NB_CHANNELS),
		       data_format='channels_last'))
	cnn.add(Activation('relu'))
	cnn.add(MaxPooling2D(pool_size=(2,2),
		             strides=2))

	cnn.add(Conv2D(filters=64,
		       kernel_size=(2,2),
		       strides=(1,1),
		       padding='valid'))
	cnn.add(Activation('relu'))
	cnn.add(MaxPooling2D(pool_size=(2,2),
		             strides=2))

	cnn.add(Flatten())        
	cnn.add(Dense(64))
	cnn.add(Activation('relu'))

	cnn.add(Dropout(0.25))

	cnn.add(Dense(1))
	cnn.add(Activation('sigmoid'))

	cnn.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	cnn.load_weights('cnn_baseline.h5')


	print(path)
	image = cv2.imread(path)   
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	letter = cnn.predict(image)

	letter_dict = {1: "A", 2: "B", 3: "C", 4: "D", 5:"del", 6: "E", 7:"F", 8:"G", 9:"H", 10: "I", 11: "J", 12: "K", 13: "L", 14:"M", 
		15:"N", 16:"", 17:"O", 18:"P", 19:"Q", 20: "R", 21:"S", 22: " ", 23: "T", 24:"U", 25:"V", 26:"W", 27:"X", 28:"Y", 29:"Z"}

	letter2 = letter.tolist()
	#print(letter_dict[letter2[0][0]]) 
	return(letter_dict[letter2[0][0]])



#temp = testNet()


