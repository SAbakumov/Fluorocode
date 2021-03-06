# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:26:23 2020

@author: Sergey
"""

import tensorflow as tf
import matplotlib as plt
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import CNNHelper.NNBuilder as nn

def LSTMAutoEncoder1D():
    input = layers.Input(shape = (None,64,1))
    x = tf.cast(input, dtype=np.float32)
    
    # x = layers.TimeDistributed(layers.BatchNormalization())(input)   
    x = layers.TimeDistributed(layers.Conv1D(64,3,padding='same',activation='elu'))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPool1D(pool_size=2))(x) 



    x =  layers.TimeDistributed(layers.Conv1D(128,3,padding='same',activation='elu'))(x)
    x =  layers.TimeDistributed( layers.BatchNormalization())(x)
    x =  layers.TimeDistributed(layers.Conv1D(128,3,padding='same',activation='elu'))(x)
    x =  layers.TimeDistributed( layers.BatchNormalization())(x)
    x =  layers.TimeDistributed(layers.MaxPool1D(pool_size=2))(x)   

    
    x =  layers.TimeDistributed(layers.Conv1D(128,3,padding='same',activation='elu'))(x)
    x =  layers.TimeDistributed( layers.BatchNormalization())(x)
    x =  layers.TimeDistributed(layers.Conv1D(128,3,padding='same',activation='elu'))(x)
    x =  layers.TimeDistributed( layers.BatchNormalization())(x)
    x =  layers.TimeDistributed(layers.MaxPool1D(pool_size=2))(x)   



    x =  layers.TimeDistributed(layers.Conv1D(256,3,padding='same',activation='elu'))(x)
    x =  layers.TimeDistributed( layers.BatchNormalization())(x)
    x =  layers.TimeDistributed(layers.Conv1D(256,3,padding='same',activation='elu'))(x)
    x =  layers.TimeDistributed( layers.BatchNormalization())(x)
    x =  layers.TimeDistributed(layers.MaxPool1D(pool_size=2))(x)   


    x =  layers.TimeDistributed(layers.Flatten())(x)
    

    x =  layers.LSTM(units =200,activation='tanh',recurrent_activation = 'sigmoid')(x)



    output = layers.Dense(1,activation =  'sigmoid')(x)


    
    CNN = tf.keras.Model(input,output,name='LSTMAutoEncoder')
    return CNN

    