
import tensorflow as tf
# import matplotlib as plt
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import CNNHelper.NNBuilder as nn


def MLPDNA(classes):
    input = layers.Input(shape = (256,1))
    x = tf.cast(input, dtype=np.float32)
    x = tf.math.divide(x,100)

    x = layers.Dense(512,activation='elu')(x)
    x = layers.Dense(512,activation='elu')(x)
    x = layers.Dense(128,activation='elu')(x)

    output = layers.Dense(units=classes, activation = 'sigmoid')(x)

    DenseDNA = tf.keras.Model(input,output,name='DenseDNA')
    return  DenseDNA 
