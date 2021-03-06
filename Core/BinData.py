# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 11:07:06 2020

@author: Boris
"""
import numpy as np

def BinData(Data,factor):
    try:
        BinnedData = np.zeros([Data.shape[0],int(np.round(Data.shape[1]/2)),Data.shape[2]])

        for i in range(0,Data.shape[0]):
            for j in range(0,int(Data.shape[1]/2)):
                # print(j*factor)
                # print(j*factor+factor)
    
                BinnedData[i,j,:] = np.sum(Data[i,j*factor:j*factor+factor])/factor
    except:
           BinnedData = np.zeros(int(np.round(Data.shape[0]/2)))

           for j in range(0,int(Data.shape[0]/2)):
                # print(j*factor)
                # print(j*factor+factor)
    
                BinnedData[j] = np.sum(Data[j*factor:j*factor+factor])/factor
            
            
    return BinnedData