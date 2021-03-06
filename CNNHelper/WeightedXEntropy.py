# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:15:58 2020

@author: Boris
"""
import sys
import numpy as np
import tensorflow as tf
from CNNHelper.CNN1D import CNN1D

from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from Core.DataHandler import DataLoader
import sklearn.model_selection as sk
import matplotlib.pyplot as plt

def weighted_categorical_crossentropy(y_true, y_pred):
    w = tf.constant([1/0.92, 1/0.04 , 1/0.04])
    cross_entropy = tf.reduce_mean( -tf.reduce_sum(w*y_true*tf.math.log(tf.clip_by_value(y_pred,1e-10,1.0)),axis = -1))
    return cross_entropy
