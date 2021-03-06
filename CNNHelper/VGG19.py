# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:58:50 2020

@author: Sergey
"""
import tensorflow as tf
import matplotlib as plt
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import CNNHelper.NNBuilder as nn

def VGG19(classes):
    classes = 3
    
    input = layers.Input(shape = (None,None,1))
    convOutput1 = nn.Blocks.ConvBlock(3,2,64,1,'relu',input)
    # convOutput1 = keras.layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size= (2,2))(convOutput1)
    # x = layers.Dropout(0.2)(x)
    
    convOutput2  = nn.Blocks.ConvBlock(3,2,128,1,'relu',x)
    # convOutput2 = keras.layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size= (2,2))(convOutput2)
    # x = layers.Dropout(0.3)(x)

    convOutput3 = nn.Blocks.ConvBlock(3,2,256,1,'relu',x)
    # convOutput3 = keras.layers.BatchNormalization()(x)
    x= layers.MaxPooling2D(pool_size= (2,2))(convOutput3)
    # x = layers.Dropout(0.4)(x)

    
    x = nn.Blocks.ConvBlock(3,4,512,1,'relu',x)
    # x = keras.layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size= (2,2))(x)
    # x = layers.Dropout(0.5)(x)

    
    # x = nn.Blocks.ConvBlock(3,4,512,1,'relu',x)
    # # x = keras.layers.BatchNormalization()(x)
    # x = layers.MaxPooling2D(pool_size= (2,2))(x)
    # # x = layers.Dropout(0.5)(x)

    # x = layers.Conv2DTranspose(256,kernel_size = (2,2), strides = (2,2))(x)
    # x = layers.Conv2D(256,kernel_size = 3,padding = 'same',activation = 'relu')(x)
    # x = keras.layers.BatchNormalization()(x)
    
    
    
    x = layers.Conv2DTranspose(256,kernel_size = (2,2), strides = (2,2))(x)
    x = layers.Conv2D(256,kernel_size = 3,padding = 'same',activation = 'relu')(x)
    x = keras.layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(256,kernel_size = (2,2), strides = (2,2))(x)
    x = layers.Conv2D(256,kernel_size = 3,padding = 'same',activation = 'relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = layers.Concatenate()([convOutput3,x])
    
    x = layers.Conv2DTranspose(128,kernel_size = (2,2), strides = (2,2))(x)
    x = layers.Conv2D(128,kernel_size = 3,padding = 'same',activation = 'relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = layers.Concatenate()([convOutput2,x])
    
    x = layers.Conv2DTranspose(64,kernel_size = (2,2), strides = (2,2))(x)
    x = layers.Conv2D(64,kernel_size = 3,padding = 'same',activation = 'relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = layers.Concatenate()([convOutput1,x])
    
    
    output = layers.Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='softmax', padding='valid', strides=(1, 1))(x)
    print(output.shape)
    VGG = tf.keras.Model(input,output,name='VGG19')

    return VGG
