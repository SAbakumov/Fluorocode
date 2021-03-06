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

def ResNet50(classes):
      input = layers.Input(shape = (256,256,1))
      x = layers.Conv2D(128,7,padding='same')(input)
      
      x = layers.BatchNormalization()(x)    

      x = layers.MaxPool2D(pool_size=(3,3),strides = 2,padding='same')(x)
      # x = layers.SpatialDropout2D(0.2)(x)

      
      x = nn.Blocks.ConvBlock( [1,3,1],  3,   [128,128,256],1, ['relu','relu',None],x)
      for i in range(0,6):
          y = nn.Blocks.ConvBlock( [1,3,1],  3,   [128,128,256],1, ['relu','relu',None],x)
          
          x = layers.Add()([x,y])
          x = layers.Activation('relu')(x)
      ConvOutput_1 = x

      x = layers.BatchNormalization()(x)    
      x = layers.MaxPool2D(pool_size=(3,3),strides = 2,padding='same')(x)
      # x = layers.SpatialDropout2D(0.3)(x)
      x = nn.Blocks.ConvBlock( [1,3,1],  3,   [128,128,512],1,['relu','relu',None],x)
      for i in range(0,15):
          y = nn.Blocks.ConvBlock( [1,3,1],  3,   [128,128,512],1,['relu','relu',None],x)
          x = layers.Add()([x,y])
          x = layers.Activation('relu')(x)
      
      ConvOutput_2 = x
    
      x = layers.BatchNormalization()(x)    
      x = layers.MaxPool2D(pool_size=(3,3),strides = 2,padding='same')(x)
      # x = layers.SpatialDropout2D(0.4)(x)

      x = nn.Blocks.ConvBlock( [1,3,1],  3,   [512,512,1024],1,['relu','relu',None],x)
      for i in range(0,5):
            y = nn.Blocks.ConvBlock( [1,3,1],  3,     [512,512,1024],1,['relu','relu',None],x)
            x = layers.Add()([x,y])
            x = layers.Activation('relu')(x) 
       # ConvOutput_3 = x
    
          
      # x = layers.MaxPool2D(pool_size=(3,3),strides = 2,padding='same')(x)
      # x = nn.Blocks.ConvBlock( [1,3,1],  3,   [512,512,2048],[1,1,1],'relu',x)    
      # for i in range(0,2):
      #     y = nn.Blocks.ConvBlock( [1,3,1],  3,  [512,512,2048],[1,1,1],'relu',x)
      #     x = layers.Add()([x,y])
      #     x = layers.Activation('relu')(x)    
      
        
      
      # x = layers.Conv2DTranspose(1024,kernel_size = (2,2), strides = (2,2))(x)
      # x = layers.Conv2D(1024,kernel_size = 3,padding = 'same',activation = 'relu')(x)
      # x = layers.Concatenate()([ConvOutput_3,x])
      
      
      x = layers.Conv2DTranspose(512,kernel_size = (2,2), strides = (2,2))(x)
      x = layers.Conv2D(512,kernel_size = 3,padding = 'same',activation = 'relu')(x)
      x = layers.BatchNormalization()(x)    
      x = layers.Concatenate()([ConvOutput_2,x])
      
      
      x = layers.Conv2DTranspose(256,kernel_size = (2,2), strides = (2,2))(x)
      x = layers.Conv2D(256,kernel_size = 3,padding = 'same',activation = 'relu')(x)
      x = layers.BatchNormalization()(x)    
      x = layers.Concatenate()([ConvOutput_1,x])
      
      
      x = layers.Conv2DTranspose(128,kernel_size = (2,2), strides = (2,2))(x)
      x = layers.Conv2D(128,kernel_size = 3,padding = 'same',activation = 'relu')(x)
      x = layers.BatchNormalization()(x)    

      
      output = layers.Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='softmax', padding='valid', strides=(1, 1))(x)
      print(output.shape)
      ResNET50 = tf.keras.Model(input,output,name='ResNET50')
      
      return ResNET50
      
      
      

      
  
