# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 20:30:11 2020

@author: Sergey
"""

import tensorflow as tf
# import matplotlib as plt
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import CNNHelper.NNBuilder as nn

def CNN1D(classes):
    input = layers.Input(shape = (256,1))
    x = tf.cast(input, dtype=np.float32)
    x = tf.math.divide(x,100)


    

    x = layers.Conv1D(64,3,padding='same',activation='elu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64,3,padding='same',activation='elu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    
    x = layers.Conv1D(128,3,padding='same',activation='elu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128,3,padding='same',activation='elu')(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Conv1D(128,3,padding='same',activation='elu')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Conv1D(128,3,padding='same',activation='elu')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    

    x = layers.Conv1D(256,3,padding='same',activation='elu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(256,3,padding='same',activation='elu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(256,3,padding='same',activation='elu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(256,3,padding='same',activation='elu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)


    x = layers.Conv1D(256,3,padding='same',activation='elu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(256,3,padding='same',activation='elu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(256,3,padding='same',activation='elu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(256,3,padding='same',activation='elu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.GlobalMaxPooling1D()(x)

    # x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(units=512, activation = 'elu')(x)
    x = layers.Dense(units=256, activation = 'elu')(x)


    output = layers.Dense(units=classes, activation = 'sigmoid')(x)
    
    CNN = tf.keras.Model(input,output,name='CNN1D')
    return CNN