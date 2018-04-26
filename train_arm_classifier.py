import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import io
import os
from PIL import Image

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from preprocess import preprocess, rgb2gray, show_image

data = []
labels = []
label_index = 0
width = 2
height = 2
scale = 2
DATAPATH = '/Users/alexchan/Documents/college/uavs/flywave/'

for label in os.listdir(DATAPATH):
    if label == 'rightup' or label == 'leftup':
        img_dir = DATAPATH + label + '/'
        count = 0
        for fn in os.listdir(img_dir):
            print(img_dir + fn)
            if fn != "_DS_Store":
                try:
                    img = cv2.imread(img_dir + fn, 1)
                    print(img.shape)
                    if width == None or height == None:
                        height = img.shape[0]
                        width = img.shape[1]
                    img = preprocess(img, scale)
                    if count == 3 or count == 100:
                        cv2.imshow("Vid", img)
                    data.append(img)
                    labels.append(label_index)
                except:
                    pass
            count += 1
        label_index += 1

print("Finished loading data")

data = np.array(data)
labels = np.array(labels)

np.save('arm_data.npy', data)
np.save('arm_labels.npy', labels)

# data = np.load('data.npy')
# labels = np.load('labels.npy')

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=6)

batch_size = 8
num_classes = 2
epochs = 12
channels = 1

img_rows, img_cols = height//scale, width//scale

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
    input_shape = (channels, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
    input_shape = (img_rows, img_cols, channels)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

from cnn_model import createModel
model = createModel(input_shape, 2)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
