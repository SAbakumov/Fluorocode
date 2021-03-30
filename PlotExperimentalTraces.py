# -*- coding: utf-8 -*-
#%%
"""
Created on Tue Mar  2 13:14:08 2021

@author: Sergey
"""

from Core.DataHandler import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random
#%%
dt = DataLoader()
x_vFp = dt.LoadTracesFromCSV(r'D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\Green\DataForGreen0-0GeneratedTraces.csv')
NumPlots = 5
plt.close('all')
for i in range(1,NumPlots+1):
   plt.subplot(NumPlots, 1,i)
   
   trc = random.choice(x_vFp)
   # trc = x_vFp[i]

   plt.plot(trc)
# %%
