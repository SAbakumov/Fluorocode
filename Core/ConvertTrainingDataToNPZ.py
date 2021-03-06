# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 17:31:49 2020

@author: Sergey
"""
import os
import numpy as np
import ReturnInputImageFromFolder as RII
import tensorflow as tf
import matplotlib.pyplot as plt


input_dir_labels = "C:/Documents/TrainingData/VHarvey/Labels"
input_dir_data = "C:\Documents\TrainingData\VHarvey\Training"
target_dir = "C:\Documents\TrainingData\VHarvey\DataForTraining"


LoadFrom = 1;
LoadTo = 256;

LoadFromToTest = 257
LoadToToTest = 287
PrefixLabel = 'mask-FOV_Crop'
PrefixInput = 'FOV_Crop'
FileFormat = '.tif'

x_train = np.empty([LoadTo-LoadFrom+1,512,512,1])
y_train = np.empty([LoadTo-LoadFrom+1,512,512,1])


y_test = np.empty([30,512,512,1])



#Load Reference labels
for file in range(LoadFrom, LoadTo):
    directoryLabel = input_dir_labels + os.path.sep + PrefixLabel + str(file) + FileFormat
    directoryInput = input_dir_data + os.path.sep + PrefixInput + str(file) + FileFormat
    
#    training_labels  = RII.ReturnInputLabelImage(directoryLabel,training_labels,file)
    image, mask  =   RII.ReturnInputAndLabelImage(directoryLabel,directoryInput)
    
    x_train[file,:,:,:] = (image*255);
    y_train[file,:,:,:] = mask;
    
    
np.savez("C:\Documents\TrainingData\VHarvey\DataForTraining.npz",training_data=x_train.astype(np.uint8),training_labels=y_train.astype(np.uint8))    
    
samples_to_predict = [];


for file in range(LoadFromToTest, LoadToToTest):
    directoryLabel = input_dir_labels + os.path.sep + PrefixLabel + str(file) + FileFormat
    directoryInput = input_dir_data + os.path.sep + PrefixInput + str(file) + FileFormat
    
#    training_labels  = RII.ReturnInputLabelImage(directoryLabel,training_labels,file)
    image, mask  =   RII.ReturnInputAndLabelImage(directoryLabel,directoryInput)
    y_test[file-LoadFromToTest,:,:,:] = ((image*255));
    
np.savez("C:\Documents\TrainingData\VHarvey\DataForTesting.npz",samples_to_predict= y_test.astype(np.uint8))    

    
#X_train = tf.convert_to_tensor(x_train)
#Y_train = tf.convert_to_tensor(y_train)

#    training_label =  np.expand_dims(training_labels,axis=2);
#    training_images = np.expand_dims(training_data,axis=2);

#print(X_train.shape, Y_train.shape)

