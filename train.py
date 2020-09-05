import os
import cv2
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import Unet
from tiny_model import TinyUnet
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K

net = TinyUnet(num_classes=2)
model = net.get_model()
fig, ax = plt.subplots(3,3, figsize=(15,15))

testing_images = ['000e218f21.png', '2c707479f9.png', '589e94265a.png']

NUM_TRAIN_IMG=100
BATCH_SIZE=16
EPOCHS=100 
DATA_DIR = 'data/images'
LABEL_DIR = 'data/masks'
MODEL_CHECKPOINT = 'checkpoints/model.weights.hdf5'
INPUT_SHAPE = (128, 128, 1)
OUTPUT_SHAPE = (128,128, 2)

if(os.path.exists(MODEL_CHECKPOINT)):
    print('[INFO] Transfer learning from old checkpoints ... ')
    model.load_weights(MODEL_CHECKPOINT)

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

        train_img.append(img)
        train_labels.append(label_img)
        print('[INFO] Reading file : ' + abs_path)
        print('[INFO]   Label path : %s ' % label_id)

train_img = np.array(train_img)
train_labels = tf.one_hot(np.array(train_labels), depth=2)
mean_iou = tf.keras.metrics.MeanIoU(num_classes=2)

callbacks = [
    EarlyStopping(patience=5, verbose=1),
    ModelCheckpoint(MODEL_CHECKPOINT, verbose=1, save_best_only=True)
]


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.fit(train_img, train_labels, epochs=EPOCHS, validation_split=0.1, batch_size=BATCH_SIZE, callbacks=callbacks)



for i, img in enumerate(testing_images):
    img_path = DATA_DIR + '/' + img
    label_path = LABEL_DIR + '/' + img

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (INPUT_SHAPE[0], INPUT_SHAPE[1]))

    label = cv2.imread(label_path)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    label = cv2.resize(label, (OUTPUT_SHAPE[0], OUTPUT_SHAPE[1]))

    prediction = model.predict(np.array([img]))[0]
    prediction = (prediction > 0.5).astype(np.uint8)[:,:,1]
    print(prediction)

    ax[i][0].imshow(img)
    ax[i][0].set_title("Input")

    ax[i][1].imshow(label)
    ax[i][1].set_title("Ground Truth")

    ax[i][2].imshow(prediction)
    ax[i][2].set_title("Prediction")

plt.show()
