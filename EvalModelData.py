# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 02:01:21 2021

@author: Boris
"""
from Core import Misc,SIMTraces
from Core.DataHandler import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as  mpl
import json
from keras.models import model_from_json
import os
import time
import sklearn.metrics
from cycler import cycler
import copy
import Core.RandomTraceGenerator as R
import matplotlib
import tensorflow as tf
import sklearn
from tensorflow.keras.models import Model


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    except RuntimeError as e:
        print(e)

matplotlib.rcParams.update({'image.cmap':'inferno'})

colors= mpl.cm.rainbow(np.linspace(0,1,7))

matplotlib.rcParams.update({'font.size': 18})


mpl.rcParams['axes.prop_cycle'] = cycler(color=colors)

Params = {"Wavelength" : 576,
           "NA" : 1.4,
           "FragmentSize" :300,
           "PixelSize" : 32.25*2,
           "ResEnhancement":1,
           "GeneratorType" : 'FromFull',
           "ArtificialDyeNumber" : 15462,
           "ArtificialGenomeLen" : 66811,
           "NumTransformations"  : [600,600],
           "StretchingFactor" :[1.72 ],
           "LowerBoundEffLabelingRate" : 0.68,
           "UpperBoundEffLabelingRate" : 0.9,
     
           "amplitude-variation":[8.55696606597531,	3.23996722003733],
           "step" :2,
           "PixelShift": 2,
           "NoiseAmp": [5],
           "GenerateFullReference" :True,
           "LocalNormWindow":0,
           "ZNorm": False,
           "Norm":  True,
           "Date" : 0,
           "Type" : "Checking",
           # "Genomes" : ['NC_000913.3'],
           "FPR": 0.9, #per kb
           "FPR2": 0.4, #per kb
            "Genomes" : ['NC_000913.3','NZ_LR735434.1','NZ_LR135297.1','NC_005139.1','NC_006177.1','NC_009663.1','NZ_CP021851.1'],
           "Random-min": 52,
           "Random-max": 210}


dt = DataLoader()
    
    

# for genome in Params["Genomes"]:
    

    
    
#     ReferenceData = []
#     EffLabeledTraces = []
#     Profiles = []
#     SIMTRC     = SIMTraces.TSIMTraces(genome,Params["StretchingFactor"],0.34,0,'TaqI',Params["PixelSize"],Params['PixelShift'],Params[ "amplitude-variation"] ,Params["FPR"],Params["FPR2"])  
#     Gauss  = Misc.GetGauss1d(Params["FragmentSize"] , Misc.FWHMtoSigma(Misc.GetFWHM(Params["Wavelength"],Params["NA"],Params["ResEnhancement"])),Params["PixelSize"] )
    
        
# x_vFp = dt.LoadTracesFromCSV(r'D:\Elizabete\2020\20201118_imaged_SIM\ZYMOPerseus\segmented_traces_averages.csv')
stretchFactors =[ 1.72]
x_vReal,lags,Strch,Direc,CorrectNum,pval  = dt.LoadMatchedTracesFromCSV(r'D:\Sergey\FluorocodeMain\FluorocodeMain\Data\Real\E-Coli-Elyra-WF\segmented_traces_averages.csv',r'D:\Sergey\FluorocodeMain\FluorocodeMain\Data\Real\E-Coli-Elyra-WF\Results\FILE1-all-reference-species-results.csv', 256)
x_vFp = dt.LoadTracesFromCSV(r'D:\Elizabete\2021\VibrioHarveiiWidefieldCirculomics-Elizabete\Circulomics\Image 4\Export\0\Traces\segmented_traces_averages.csv')
x_vReal = dt.LoadTracesFromCSV(r'D:\Sergey\FluorocodeMain\FluorocodeMain\Data\Real\E-Coli-Elyra-WF\segmented_traces_averages.csv')

PValFalse, PValTrue  = dt.LoadPValsFromCSV(r'D:\Elizabete\2021\VibrioHarveiiWidefieldCirculomics-Elizabete\Circulomics\Image 4\Export\0\Traces\segmented_traces_averages.csv',r'D:\Elizabete\2021\VibrioHarveiiWidefieldCirculomics-Elizabete\Circulomics\Image 4\VHarvey\FILE1-all-reference-species-results.csv', 256,['CP000790.1','CP000789.1','CP000791.1'])

IdealProfs = []  
pval = [float(p) for p in pval]

PValFalse = np.array(PValFalse)



x_vAllTP1 = []
x_vAllTP2 = []

x_vAllFP1 = []
x_vAllFP2 = []
x_vRealStr = []
for i in range(0, len(x_vReal)): 
    if len(x_vReal[i])>256:
        x_vRealStr.append([x_vReal[i],i] )
                          
x_vFp = [x for x in x_vFp if len(x)>256 ]
pvalTP = np.array([pval[int(x[1])] for x in x_vRealStr])
x_vReal = [x[0] for x in x_vRealStr]

positionsTP = [np.random.randint(0,len(x)-256) for x in x_vReal]
positionsFP = [np.random.randint(0,len(x)-256) for x in x_vFp]

for stretch in stretchFactors:
    print(stretch)
    x_VFP=[]
    x_VTP=[]

    x_vRealFlipped = []
    x_vRealFlippedTP = []
    x_vFP1 = []
    x_vFP2 = []
    x_vTP1 = []
    x_vTP2 = []
    for i in range(0,467):
        
            x_VFP.append(Misc.GetLocalNormFromPars( x_vFp[i].astype(np.float32),stretch,0.34,64.5,10000)   )
            x_vRealFlipped.append(Misc.GetLocalNormFromPars( np.flip(x_vFp[i].astype(np.float32)),stretch,0.34,64.5,10000)   )
            pos = np.random.randint(0,len(x_VFP[-1])-256)
            frag = np.reshape(x_VFP[-1][positionsFP[i] :positionsFP[i] +256],[256,1])
            frag_flipped = np.reshape(x_vRealFlipped[-1][positionsFP[i] :positionsFP[i]+256],[256,1])
            x_vFP1.append(frag)
            x_vFP2.append(frag_flipped)

    for i in range(0,len(x_vReal)):
        # if pvalTP[i]<=0.001:
            x_VTP.append( Misc.GetLocalNormFromPars(  x_vReal[i].astype(np.float32),stretch,0.34,64.5,10000))
            x_vRealFlippedTP.append( Misc.GetLocalNormFromPars(  np.flip(x_vReal[i].astype(np.float32)),stretch,0.34,64.5,10000))
            pos = np.random.randint(0,len(x_vReal[i])-256)
            frag = np.reshape(x_VTP[-1][positionsTP[i] :positionsTP[i]+256],[256,1])
            frag_flipped = np.reshape(  x_vRealFlippedTP[-1][positionsTP[i] :positionsTP[i]+256],[256,1])
            x_vTP1.append(frag)
            x_vTP2.append(frag_flipped)
            

    
    x_vAllTP1.append(np.array(x_vTP1))
    x_vAllTP2.append(np.array(x_vTP2))
    
    x_vAllFP1.append(np.array(x_vFP1))
    x_vAllFP2.append(np.array(x_vFP2))





        
    



pth = r'D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\StoredModels\2021-03-08\Training_1'


json_file = open(os.path.join(pth ,'model-Architecture.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()

json_file = open(os.path.join(pth ,'logs.json'), 'r')
log = json_file.read()
json_file.close()
log = json.loads(log)
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(os.path.join(pth, 'modelBestLoss.hdf5' ))

model = loaded_model

x_vAllTP1 = [model.predict(x) for x in x_vAllTP1 ]
x_vAllTP2 = [model.predict(x) for x in x_vAllTP2 ]

x_vAllFP1 = [model.predict(x) for x in x_vAllFP1 ]
x_vAllFP2 = [model.predict(x) for x in x_vAllFP2 ]
x_vALL = []
x_vALLFP = []

for i in range(0,len(x_vAllTP1)):
    x_vALL.append(np.concatenate([x_vAllTP1[i],x_vAllTP2[i]],axis=1))
    x_vALLFP.append(np.concatenate([x_vAllFP1[i],x_vAllFP2[i]],axis=1))
    
for i in range(0,len(x_vALL)):
    x_vALL[i] = np.max(x_vALL[i],axis=1)
    x_vALLFP[i] = np.max(x_vALLFP[i],axis=1)
    
x_vALL = np.transpose(np.array(x_vALL))
x_vALLFP = np.transpose(np.array(x_vALLFP))

x_vALL = np.max(x_vALL, axis= 1)
x_vALLFP = np.max(x_vALLFP, axis= 1)


y_true = np.ones(x_vALL.shape)
y_false = np.zeros(x_vALLFP.shape)
y_true = np.concatenate([y_true,y_false])
y_pred = np.concatenate([x_vALL,x_vALLFP])
fpr,tpr,ths = sklearn.metrics.roc_curve(y_true,y_pred)
print(sklearn.metrics.auc(fpr,tpr))
plt.plot(fpr,tpr)

plt.figure()
