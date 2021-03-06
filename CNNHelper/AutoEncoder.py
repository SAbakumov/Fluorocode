# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 14:46:21 2020

@author: Boris
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 20:30:11 2020

@author: Sergey
"""

import tensorflow as tf
import matplotlib as plt
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import CNNHelper.NNBuilder as nn

def AutoEncoder1D():
    input = layers.Input(shape = (256,1))
     
    # x = layers.TimeDistributed(layers.Conv1D(512,7,padding='same',activation='elu'))(input)
    # x = layers.TimeDistributed(layers.BatchNormalization())(x)
    
    x = layers.Conv1D(1024,7,padding='same',activation='elu')(input)
    x = layers.BatchNormalization()(x)
    # x = layers.TimeDistributed(layers.Conv1D(256,7,padding='same',activation='relu'))(x)
    # x = layers.TimeDistributed(layers.BatchNormalization())(x)
    y1 = layers.MaxPool1D(pool_size=2)(x) 



    x =  layers.Conv1D(256,3,padding='same',activation='elu')(y1)
    x =  layers.BatchNormalization()(x)
    x =  layers.Conv1D(256,3,padding='same',activation='elu')(x)
    x =  layers.BatchNormalization()(x)
    # x =  layers.TimeDistributed(layers.Conv1D(256,3,padding='same',activation='relu'))(x)
    # x =  layers.TimeDistributed( layers.BatchNormalization())(x)
    # x =  layers.Add()([x,y1])
    # x =  layers.Activation('elu')(x)
    y2 = layers.MaxPool1D(pool_size=2)(x)   

    
    x =  layers.Conv1D(256,3,padding='same',activation='elu')(y2)
    x =  layers.BatchNormalization()(x)
    x =  layers.Conv1D(256,3,padding='same',activation='elu')(x)
    x =  layers.BatchNormalization()(x)
    # x =  layers.Add()([x,y2])
    # x =  layers.Activation('elu')(x)
    y3 = layers.MaxPool1D(pool_size=2)(x)   

    x =  layers.Conv1D(512,3,padding='same',activation='elu')(y3)
    x =  layers.BatchNormalization()(x)
    x =  layers.Conv1D(512,3,padding='same',activation='elu')(x)
    x =  layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x =  layers.Dense(512, activation='elu')(x)
    
    x =  layers.Dense(128, activation='relu')(x)
    output = layers.Reshape(target_shape=[128,1])(x)
  
    # output = layers.Conv1D(filters =1,kernel_size = 1,padding='causal', activation = 'elu')(x)
    CNN = tf.keras.Model(input,output,name='AutoEncoder')
    return CNN

    