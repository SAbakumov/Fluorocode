# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 09:13:07 2020

@author: Sergey
"""

import tensorflow as tf
import matplotlib as plt
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import CNNHelper.NNBuilder as nn

def ResNet50(classes,inputsize):
    input = layers.Input(shape = (inputsize,1))
    x = layers.Conv1D(128,7,padding='same')(input)
    
    x = layers.BatchNormalization()(x)    

    x = layers.MaxPool1D(pool_size=3,strides = 2,padding='same')(x)
    # x = layers.SpatialDropout2D(0.2)(x)

    
    x = nn.Blocks.ConvBlock( [1,3,1],  3,   [32,32,64],1, ['elu','elu',None],x)
    for i in range(0,6):
        y = nn.Blocks.ConvBlock( [1,3,1],  3,   [32,32,64],1, ['elu','elu',None],x)
        
        x = layers.Add()([x,y])
        x = layers.Activation('relu')(x)
    ConvOutput_1 = x

    x = layers.BatchNormalization()(x)    
    x = layers.MaxPool1D(pool_size=3,strides = 2,padding='same')(x)
    # x = layers.SpatialDropout2D(0.3)(x)
    x = nn.Blocks.ConvBlock( [1,3,1],  3,   [64,64,128],1,['elu','elu',None],x)
    for i in range(0,15):
        y = nn.Blocks.ConvBlock( [1,3,1],  3,  [64,64,128],1,['elu','elu',None],x)
        x = layers.Add()([x,y])
        x = layers.Activation('relu')(x)
    
    ConvOutput_2 = x

    x = layers.BatchNormalization()(x)    
    x = layers.MaxPool1D(pool_size=3,strides = 2,padding='same')(x)
    # x = layers.SpatialDropout2D(0.4)(x)

    x = nn.Blocks.ConvBlock( [1,3,1],  3,   [128,128,256],1,['elu','elu',None],x)
    for i in range(0,5):
        y = nn.Blocks.ConvBlock( [1,3,1],  3,     [128,128,256],1,['elu','elu',None],x)
        x = layers.Add()([x,y])
        x = layers.Activation('relu')(x) 
    # ConvOutput_3 = x

    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.GlobalMaxPooling1D()(x)

    # x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(units=512, activation = 'elu')(x)
    x = layers.Dense(units=256, activation = 'elu')(x)
    # x = layers.Dense(units=128, activation = 'elu')(x)
    # x = layers.Dense(units=64, activation = 'elu')(x)

    output = layers.Dense(units=classes, activation = 'softmax')(x)

    ResNET50 = tf.keras.Model(input,output,name='ResNET50')
    
    return ResNET50
    
      
      

      
  
