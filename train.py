import numpy as np
import tensorflow as tf

from model import Unet
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

net = Unet()
model = net.get_model()

print(model.summary())


