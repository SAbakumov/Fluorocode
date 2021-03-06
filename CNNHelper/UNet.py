# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 11:46:56 2020

@author: Boris
"""

import tensorflow as tf
import matplotlib as plt
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import CNNHelper.NNBuilder as nn

def UNET(classes):
      input = layers.Input(shape = (512,512,1))
      
      x = nn.ConvBlock(3,2,64,1,'relu',input)
      conv_1 = layers.BatchNormalization()(x)
      x = layers.MaxPool2D(pool_size= (2,2))(conv_1)
      
      
      x = nn.ConvBlock(3,2,128,1,'relu',input)
      conv_2 = layers.BatchNormalization()(x)
      x = layers.MaxPool2D(pool_size= (2,2))(conv_2)
      
      x = nn.ConvBlock(3,2,256,1,'relu',input)
      conv_3 = layers.BatchNormalization()(x)
      x = layers.MaxPool2D(pool_size= (2,2))(conv_3)
      
      
      x = nn.ConvBlock(3,2,512,1,'relu',input)
      conv_4 = layers.BatchNormalization()(x)
      x = layers.MaxPool2D(pool_size= (2,2))(conv_4)
      
      x = nn.ConvBlock(3,2,1024,1,'relu',input)
      x = layers.BatchNormalization()(x)

      x = layers.Conv2DTranspose(512,kernel_size = (2,2), strides = (2,2))(x)
      x = layers.Conv2D(512,kernel_size = 3,padding = 'same',activation = 'relu')(x)
      x = layers.Concatenate()([conv_4,x])
      
      
      x = layers.Conv2DTranspose(256,kernel_size = (2,2), strides = (2,2))(x)
      x = layers.Conv2D(256,kernel_size = 3,padding = 'same',activation = 'relu')(x)
      x = layers.Concatenate()([conv_3,x])

      x = layers.Conv2DTranspose(128,kernel_size = (2,2), strides = (2,2))(x)
      x = layers.Conv2D(128,kernel_size = 3,padding = 'same',activation = 'relu')(x)
      x = layers.Concatenate()([conv_2,x])      
      
      
      x = layers.Conv2DTranspose(64,kernel_size = (2,2), strides = (2,2))(x)
      x = layers.Conv2D(64,kernel_size = 3,padding = 'same',activation = 'relu')(x)
      x = layers.Concatenate()([conv_1,x])   
      
      
      output = layers.Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='softmax', padding='valid', strides=(1, 1))(x)
      print(output.shape)
      Unet = tf.keras.Model(input,output,name='U-Net')

      return Unet 