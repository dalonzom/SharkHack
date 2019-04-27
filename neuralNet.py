import tensorflow as tf
from tensorflow.keras import layers

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import sys
import glob 
import os


import time

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



train_datagen = ImageDataGenerator(
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    trainPath,
    target_size=(IMG_SIZE,IMG_SIZE),
    class_mode='binary',
    batch_size = BATCH_SIZE)


validation_generator = validation_datagen.flow_from_directory(
    testPath,
    target_size=(IMG_SIZE,IMG_SIZE),
    class_mode='binary',
    batch_size = BATCH_SIZE)


start = time.time()
cnn.fit_generator(
    train_generator,
    steps_per_epoch=NB_TRAIN_IMG//BATCH_SIZE,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=NB_VALID_IMG//BATCH_SIZE)
end = time.time()
print('Processing time:',(end - start)/60)

cnn.save_weights('cnn_baseline.h5')


