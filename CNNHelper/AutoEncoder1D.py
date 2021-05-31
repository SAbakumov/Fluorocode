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


class AutoEncoder1D():
    def __init__(self):
        input = layers.Input(shape = (None,1))
        x = layers.Conv1D(16,3,strides=1,padding='same',activation='elu')(input)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(16,3,strides=1,padding='same',activation='elu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool1D(pool_size=2)(x)

        x = layers.Conv1D(32,3,strides=1,padding='same',activation='elu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(32,3,strides=1,padding='same',activation='elu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool1D(pool_size=2)(x)


        x = layers.Conv1D(64,3,strides=1,padding='same',activation='elu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(64,3,strides=1,padding='same',activation='elu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool1D(pool_size=2)(x)


        x = layers.Conv1DTranspose(64,3,strides=2,activation='elu',padding='same')(x)
        x = layers.Conv1D(32,3,strides=1,padding='same',activation='elu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(32,3,strides=1,padding='same',activation='elu')(x)
        x = layers.BatchNormalization()(x)


        x = layers.Conv1DTranspose(32,3,strides=2,activation='elu',padding='same')(x)
        x = layers.Conv1D(16,3,strides=1,padding='same',activation='elu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(16,3,strides=1,padding='same',activation='elu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1DTranspose(16,3,strides=2,activation='elu',padding='same')(x)

        output = layers.Conv1D(1,1,activation='relu')(x)
        self.Autoencoder = tf.keras.Model(input,output,name='Autoencoder')

if __name__=='__main__':
    net = AutoEncoder1D()
    net.Autoencoder.summary()
