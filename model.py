import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

class Unet(object):
    def __init__(self, input_shape=(572,572,1)):
        self.input_shape=input_shape

    def get_cropping_layer(self,source_shape, target_shape):
        source_width = source_shape[2]
        target_width = target_shape[2]

        source_height = source_shape[1]
        target_height = target_shape[1]

        top_bottom_crop = int((source_height - target_height)/2)
        left_right_crop = int((source_width - target_width)/2)

        return Cropping2D(cropping=((top_bottom_crop, top_bottom_crop), (left_right_crop, left_right_crop)))

    def get_model(self, num_classes=2):
        ### Contracting path ###
        inputs = Input(shape=self.input_shape)
        conv_1_1 = Conv2D(64, kernel_size=(3,3), activation='relu', name='conv_1_1')(inputs)
        conv_1_2 = Conv2D(64, kernel_size=(3,3), activation='relu', name='conv_1_2')(conv_1_1)
        pool_1 = MaxPooling2D(pool_size=(2,2))(conv_1_2)

        conv_2_1 = Conv2D(128, kernel_size=(3,3), activation='relu', name='conv_2_1')(pool_1)
        conv_2_2 = Conv2D(128, kernel_size=(3,3), activation='relu', name='conv_2_2')(conv_2_1)
        pool_2 = MaxPooling2D(pool_size=(2,2))(conv_2_2)

        conv_3_1 = Conv2D(256, kernel_size=(3,3), activation='relu', name='conv_3_1')(pool_2)
        conv_3_2 = Conv2D(256, kernel_size=(3,3), activation='relu', name='conv_3_2')(conv_3_1)
        pool_3 = MaxPooling2D(pool_size=(2,2))(conv_3_2)

        conv_4_1 = Conv2D(512, kernel_size=(3,3), activation='relu', name='conv_4_1')(pool_3)
        conv_4_2 = Conv2D(512, kernel_size=(3,3), activation='relu', name='conv_4_2')(conv_4_1)
        pool_4 = MaxPooling2D(pool_size=(2,2))(conv_4_2)

        conv_5_1 = Conv2D(1024, kernel_size=(3,3), activation='relu', name='conv_5_1')(pool_4)
        conv_5_2 = Conv2D(1024, kernel_size=(3,3), activation='relu', name='conv_5_2')(conv_5_1)
        
        up_1 = UpSampling2D((2,2))(conv_5_2)
        conv_up_1 = Conv2D(512, kernel_size=(1,1), activation='relu', name='conv_up_1')(up_1)
        crop_conv_4 = self.get_cropping_layer(conv_4_2.shape, up_1.shape)(conv_4_2)
        merge_1 = tf.keras.layers.concatenate((crop_conv_4, conv_up_1), axis=3)

        conv_6_1 = Conv2D(512, kernel_size=(3,3), activation='relu', name='conv_6_1')(merge_1)
        conv_6_2 = Conv2D(512, kernel_size=(3,3), activation='relu', name='conv_6_2')(conv_6_1)

        up_2 = UpSampling2D((2,2))(conv_6_2)
        conv_up_2 = Conv2D(256, kernel_size=(1,1), activation='relu', name='conv_up_2')(up_2)
        crop_conv_3 = self.get_cropping_layer(conv_3_2.shape, up_2.shape)(conv_3_2)
        merge_2 = tf.keras.layers.concatenate((crop_conv_3, conv_up_2), axis=3)

        conv_7_1 = Conv2D(256, kernel_size=(3,3), activation='relu', name='conv_7_1')(merge_2)
        conv_7_2 = Conv2D(256, kernel_size=(3,3), activation='relu', name='conv_7_2')(conv_7_1)

        up_3 = UpSampling2D((2,2))(conv_7_2)
        conv_up_3 = Conv2D(128, kernel_size=(1,1), activation='relu', name='conv_up_3')(up_3)
        crop_conv_2 = self.get_cropping_layer(conv_2_2.shape, up_3.shape)(conv_2_2)
        merge_3 = tf.keras.layers.concatenate((crop_conv_2, conv_up_3), axis=3)

        conv_8_1 = Conv2D(128, kernel_size=(3,3), activation='relu', name='conv_8_1')(merge_3)
        conv_8_2 = Conv2D(128, kernel_size=(3,3), activation='relu', name='conv_8_2')(conv_8_1)

        up_4 = UpSampling2D((2,2))(conv_8_2)
        conv_up_4 = Conv2D(64, kernel_size=(1,1), activation='relu', name='conv_up_4')(up_4)
        crop_conv_1 = self.get_cropping_layer(conv_1_2.shape, up_4.shape)(conv_1_2)
        merge_4 = tf.keras.layers.concatenate((crop_conv_1, conv_up_4), axis=3)

        conv_9_1 = Conv2D(64, kernel_size=(3,3), activation='relu', name='conv_9_1')(merge_4)
        conv_9_2 = Conv2D(64, kernel_size=(3,3), activation='relu', name='conv_9_2')(conv_9_1)
        out = Conv2D(num_classes, kernel_size=(1,1), activation='relu', name='output')(conv_9_2)

        return Model(inputs, out)
