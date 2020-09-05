import os
import cv2
import glob
import numpy as np
import tensorflow as tf

from model import Unet
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

net = Unet()
model = net.get_model(num_classes=3)

NUM_TRAIN_IMG=1
DATA_DIR = 'data/images'
LABEL_DIR = 'data/masks'
INPUT_SHAPE = (572, 572, 1)
OUTPUT_SHAPE = (388,388, 1)

print(model.summary())

train_img    = []
train_labels = []
print('[INFO] Loading image data and labels ...')
for (dir, dirs, files) in os.walk(DATA_DIR):
    for i, file in enumerate(files):
        if(i >= NUM_TRAIN_IMG):
            break

        abs_path = dir + '/' + file
        label_id = LABEL_DIR + '/' + file.split('.')[0] + '.png'

        img = cv2.imread(abs_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (INPUT_SHAPE[0], INPUT_SHAPE[1]))

        label_img = cv2.imread(label_id)
        label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)
        label_img = cv2.resize(label_img, (OUTPUT_SHAPE[0], OUTPUT_SHAPE[1]))
        label_img[label_img > 0] = 1

        if(i == 0):
            cv2.imshow('Sample input', img)
            cv2.imshow('Sample output', label_img)
            key = cv2.waitKey(0)

        train_img.append(img)
        train_labels.append(label_img)
        print('[INFO] Reading file : ' + abs_path)
        print('[INFO]   Label path : %s ' % label_id)

train_img = np.array(train_img)
train_labels = np.array(train_labels)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_img, train_labels, epochs=10)
