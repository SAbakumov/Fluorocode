
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 19:59:10 2020

@author: Sergey
"""
import sys,os 
sys.path.insert(1, os.path.join(os.path.dirname(__file__),"Core"))


import Misc as Misc
import SIMTraces as SIMTraces
import numpy as np
import time
import os

import shutil
import json
import random
import csv

from Core.DataHandler import DataConverter
from Core.DataHandler import DataLoader
from datetime import date
import multiprocessing
import Core.TraceGenerator as TraceGenerator



def ConcatToCsv(path,dttype,dt):
    X_Data ,Y_Data,Label_Data, pos = dt.BatchLoadTrainingData(os.path.join( path))
    with open(os.path.join(path, dttype+'-Data.csv'), 'w',newline='') as f:
        write = csv.writer(f) 
        for row in X_Data:
            data = row.flatten()
            if not np.all(data==0):
                write.writerow(row.flatten()) 

def GenTraces(TraceGen, genome, transform, Params):        
    if genome!='Random':
        counts, traces = TraceGen.ObtainTraces(transform, genome)
        
    elif genome == 'Random':
        # counts, traces = TraceGen.ObtainRandomTraces(Params["Random-max"],Params["Random-min"],20000,genome,transform)
        counts, traces = TraceGen.ObtainRandomTraces(Params["Random-max"],Params["Random-min"],30000,genome,transform)

    return counts , traces



def CallTraceGeneration(Params):

    # np.random.seed(seed=44864)
    # Params["Lags"] = np.random.choice([x for x in range(0,25000)],400).tolist()
      
    if __name__ == '__main__':
    
        t = time.time()

        
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        savedir = Misc.GetModelSavePath(ROOT_DIR,str(date.today()))
        DataSaveDir = os.path.join(ROOT_DIR, "Data")
        if not os.path.exists(os.path.join(DataSaveDir, Params["Type"] )):
            os.makedirs(os.path.join(DataSaveDir, Params["Type"] ))

        Misc.EmptyDataFolder(os.path.join( DataSaveDir,Params["Type"]))
        Misc.WriteDataParams( os.path.join( DataSaveDir,Params["Type"]),Params)


        AllCounts =[]
        Dt = DataConverter()
        Ds = DataLoader()

        Dt.ShuffleData =Params["ShuffleData"]
        Ds.ShuffleData =Params["ShuffleData"]

        if "Database" in Params.keys():
            for  genome in os.listdir(Params["Database"]):
                Params["Genomes"].append(genome)


        for genome in Params["Genomes"]:
            
            SIMTRC     = SIMTraces.TSIMTraces(genome,Params["StretchingFactor"],0.34,0,Params["Enzyme"],Params["PixelSize"],Params['PixelShift'],Params[ "amplitude_variation"] ,Params["FPR"],Params["FPR2"],Params["FragmentSize"])
            if "Database" in Params.keys():
                SIMTRC.set_db_path(Params["Database"])
            Gauss      = Misc.GetGauss1d(Params["FragmentSize"] , Misc.FWHMtoSigma(Misc.GetFWHM(Params["Wavelength"],Params["NA"],Params["ResEnhancement"])),Params["PixelSize"] )
            [Map,ReCutsInPx]  = SIMTRC.GetGenome(Params,genome)
            TraceGen   = TraceGenerator.TraceGenerator(SIMTRC, ReCutsInPx,Gauss,[],Ds, Dt,Params,DataSaveDir)
            TraceGen.SaveMap(Map,genome)
            

            
            arg = [tuple([TraceGen,genome, t,Params]) for t in range(Params["NumTransformations"][Params["Genomes"].index(genome)]) ]
            pool = multiprocessing.Pool(processes=24)
            return_vals = pool.starmap(GenTraces, arg)
            pool.close()
            pool.join()   

            TraceGen.PlotNTraces([x[1] for x  in return_vals],genome)
            AllCounts= AllCounts+[x[0] for x  in return_vals]
            print('done')
            
        
            np.savez(os.path.join( DataSaveDir,Params["Type"],"NumberOfTraces.npz"),NumberOfTraces=np.sum(np.array(AllCounts)))
            print(str(time.time()-t) + " elapsed for generation" )

        if Params["ConcatToCsv"]:
            ConcatToCsv(os.path.join( DataSaveDir,Params["Type"]),Params["Type"],Ds)
        
           
    
########### USER INPUT ############
    

Params = {"Wavelength" : 586,
        "NA" : 1.2,
        "FragmentSize" :412,
        "PixelSize" : 31.3*2,
        "ResEnhancement":1,
        "FromLags" :False,
        "ShuffleData":False,
        "Lags":[],
        "Enzyme" : 'TaqI',
        "NumTransformations"  :[50],
        "StretchingFactor" :[1.7],
        "LowerBoundEffLabelingRate" : 0.75,
        "UpperBoundEffLabelingRate" : 0.95,
        "amplitude_variation":[8.55696606597531,	3.23996722003733],
        "step" :3,
        "PixelShift":1.2,
        "NoiseAmp": [0.3],
        "GenerateFullReference" :True,
        "LocalNormWindow":10000,
        "ZNorm": False,
        "Norm":  False,
        "Date" : str(date.today()),
        "Type" : "Training",
        # "Genomes" : ['Random','NC_000913.3','CP034237.1','NC_007795.1','NC_003197.2','NC_000964.3','NZ_CP045605.1','LR215978.1','NC_004567','NC_004307','CR626927.1'],
        "Genomes" : ['Random'],
        "Database": r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\DataBases\GUT",
        # "Classes": [1,0],
        "FPR": 0.7, #per kb 0.5
        "FPR2": 0.2, #per kb 0.2
        "Random-min": 1.2, #per kb
        "Random-max": 5.8, #per kb
        # "Random-min": 10,
        # "Random-max": 15,
        "SaveFormatAsCSV":False,
        "ConcatToCsv":False}    


DataTypes = ["Training","Validation"]
# DataTypes = ["Testing"]

TrainNumTrans = (65*np.ones(len(os.listdir(Params["Database"]))+len(Params["Genomes"]),dtype=np.int32)).tolist()
ValNumTrans = (5*np.ones(len(os.listdir(Params["Database"]))+len(Params["Genomes"]),dtype=np.int32)).tolist()

# DataTypes = ["AutoencoderTrain","AutoencoderVal"]
NumTransforms = [TrainNumTrans , ValNumTrans]

# NumTransforms = [[65,65,65,65,65,65,65,65,65,65,65],[5,5,5,5,5,5,5,5,5,5,5]]
# NumTransforms = [[5],[2]]

# DataTypes = ["Testing"]
# NumTransforms = [[180,180],[20,20]]
# NumTransforms = [[1,1]]

for i in range(0,len(DataTypes)):
    Params["Type"]   =DataTypes[i]
    Params["NumTransformations"] =  NumTransforms[i]
    # Params["Lags"] =( np.random.randint(0,68000,5000)).tolist()
    CallTraceGeneration(Params)
   
###############################
    
    
    
    
    
    
