# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 09:58:06 2020

@author: Sergey
"""

import tensorflow as tf
# import matplotlib as plt
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class Blocks:
        
        
    def ConvBlock(kernelSz,NumLayers,NumKernels,Strides,Activations,x):
        # self.kernelSz = kernelSz
        # self.NumLayers = NumLayers
        # self.NumKernels = NumKernels
        # self.Strides = Strides
        # self.Activations = Activations
        
        
        for i in range(0,NumLayers):
            KernelNumber = Blocks.GetProperty(NumKernels,i)
            Stride = Blocks.GetProperty(Strides,i)
            Activation = Blocks.GetProperty(Activations,i)
            KernelSize = Blocks.GetProperty(kernelSz,i)

            

            x = layers.Conv2D(KernelNumber,KernelSize, Stride , padding = 'same')(x)    

            # x = layers.Conv2D(KernelNumber,KernelSize, Stride ,kernel_regularizer=tf.keras.regularizers.l2(0.001), padding = 'same')(x)    
            if Activation!=None:
                x=layers.Activation(Activation)(x)
            x = layers.BatchNormalization()(x)

    
        return x
    
    
    
    def GetProperty(Prop,i):
        if type(Prop)==list:
            Val = Prop[i]
        else:
            Val = Prop
        return Val
                
# inputs = keras.Input(shape=(28, 28, 1))
# outputs = Blocks.ConvBlock(3,4,64,1,'relu',inputs)                
# autoencoder = keras.Model(inputs, outputs, name="autoencoder")
                    
           
        
    
