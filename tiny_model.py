import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

class TinyUnet(object):
    def __init__(self, input_shape=(128,128,1), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def get_crop_layer(self, src, target):
        src_w = src.shape[2]
        target_w = target.shape[2]

        src_h = src.shape[1]
        target_h = target.shape[1]

        crop_h = int((src_h - target_h)/2)

    def get_model(self):
        inputs = Input(shape=self.input_shape) ### 128 x 128 x 1 ###

        ''' Contracting path '''
        ### 128 x 128 x 8 ###
        conv_1 = Conv2D(8, kernel_size=(3,3), activation='relu', padding='same')(inputs)
        ### 128 x 128 x 8 ###
        conv_1 = Conv2D(8, kernel_size=(3,3), activation='relu', padding='same')(conv_1)
        ### 64 x 64 x 8 ###
        pool_1 = MaxPooling2D(pool_size=(2,2))(conv_1)

        ### 64 x 64 x 16 ###
        conv_2 = Conv2D(16, kernel_size=(3,3), activation='relu', padding='same')(pool_1)
        ### 64 x 64 x 16 ###
        conv_2 = Conv2D(16, kernel_size=(3,3), activation='relu', padding='same')(conv_2)
        ### 32 x 32 x 16 ###
        pool_2 = MaxPooling2D(pool_size=(2,2))(conv_2)

        ### 32 x 32 x 32 ###
        conv_3 = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(pool_2)
        ### 32 x 32 x 32 ###
        conv_3 = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(conv_3)
        ### 16 x 16 x 32 ###
        pool_3 = MaxPooling2D(pool_size=(2,2))(conv_3)

        ### 16 x 16 x 64 ###
        conv_4 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(pool_3)
        ### 16 x 16 x 64 ###
        conv_4 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(conv_4)
        ### 8 x 8 x 64 ###
        pool_4 = MaxPooling2D(pool_size=(2,2))(conv_4)

        ### 8 x 8 x 128 ###
        conv_5 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(pool_4)
        ### 8 x 8 x 128 ###
        conv_5 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(conv_5)

        ''' Expansive path '''
        ### 16 x 16 x 64 ###
        up_1 = Conv2DTranspose(64, kernel_size=(2,2), strides=(2,2), padding='same')(conv_5)
        ### 16 x 16 x 128 ###
        concat_1 = tf.keras.layers.concatenate((conv_4, up_1), axis=3)
        ### 16 x 16 x 64 ###
        conv_6 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(concat_1)
        conv_6 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(conv_6)

        ### 32 x 32 x 32 ###
        up_2 = Conv2DTranspose(32, kernel_size=(2,2), strides=(2,2), padding='same')(conv_6)
        ### 32 x 32 x 64 ###
        concat_2 = tf.keras.layers.concatenate((conv_3, up_2), axis=3)
        ### 32 x 32 x 32 ###
        conv_7 = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(concat_2)
        conv_7 = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(conv_7)

        ### 64 x 64 x 16 ###
        up_3 = Conv2DTranspose(16, kernel_size=(2,2), strides=(2,2), padding='same')(conv_7)
        ### 64 x 64 x 32 ###
        concat_3 = tf.keras.layers.concatenate((conv_2, up_3), axis=3)
        ### 64 x 64 x 16 ###
        conv_8 = Conv2D(16, kernel_size=(3,3), activation='relu', padding='same')(concat_3)
        conv_8 = Conv2D(16, kernel_size=(3,3), activation='relu', padding='same')(conv_8)

        ### 128 x 128 x 8 ###
        up_4 = Conv2DTranspose(8, kernel_size=(3,3), strides=(2,2), padding='same')(conv_8)
        ### 128 x 128 x 16 ###
        concat_4 = tf.keras.layers.concatenate((conv_1, up_4), axis=3)
        ### 128 x 128 x 8 ###
        conv_9 = Conv2D(8, kernel_size=(3,3), activation='relu', padding='same')(concat_4)
        conv_9 = Conv2D(8, kernel_size=(3,3), activation='relu', padding='same')(conv_9)

        outputs = Conv2D(self.num_classes, kernel_size=(1,1), activation='sigmoid')(conv_9)

        return Model(inputs, outputs)
