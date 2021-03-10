# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 12:43:02 2021

@author: Boris
"""

from Core.DataHandler import DataLoader
import numpy as np
# import matplotlib.pyplot as plt
import csv
import os
import shutil

GreenCurrentPath = r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\Green"
GreenDestinationPath = r"D:\Sergey\FluorocodeMain\dna-matching-channels\MultiColorData\Data\0"
files = os.listdir(GreenCurrentPath)
for file in files:
    if '.npz' and 'Data' in file:
        shutil.copy(os.path.join(GreenCurrentPath, file), GreenDestinationPath)

RedCurrentPath = r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\Red"
RedDestinationPath = r"D:\Sergey\FluorocodeMain\dna-matching-channels\MultiColorData\Data\1"
files = os.listdir(RedCurrentPath)
for file in files:
    if '.npz' and 'Data' in file:
        shutil.copy(os.path.join(RedCurrentPath,file), RedDestinationPath)





