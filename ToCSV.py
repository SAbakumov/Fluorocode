# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 12:43:02 2021

@author: Boris
"""

from Core.DataHandler import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

dt = DataLoader()
mypath = r"D:\Sergey\FluorocodeMain\FluorocodeMain\Data\Green\Map\traces-Red.csv"
X_Data ,Y_Data,Label_Data, pos  = dt.BatchLoadTrainingData("D:\Sergey\FluorocodeMain\FluorocodeMain\Data\Red")
X_Data = np.squeeze(X_Data)
with open(os.path.join(mypath), 'w',newline='' ) as f: 
  
    # using csv.writer method from CSV package 
    write = csv.writer(f) 
    X_Data = X_Data.tolist()
    for x in X_Data:
        write.writerow(x) 