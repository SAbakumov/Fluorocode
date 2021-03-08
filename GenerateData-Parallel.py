
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 19:59:10 2020

@author: Sergey
"""
import Core.Misc as Misc
import Core.SIMTraces as SIMTraces
import numpy as np
import time
import os

from Core.DataHandler import DataConverter
from Core.DataHandler import DataLoader
from datetime import date
import multiprocessing
import Core.TraceGenerator as TraceGenerator



def GenTraces(TraceGen, genome, transform, Params):        
    if genome!='Random':
        counts = TraceGen.ObtainTraces(transform, genome)
        
    elif genome == 'Random':
        counts = TraceGen.ObtainRandomTraces(Params["Random-max"],Params["Random-min"],40000,genome,transform)
    return counts



def CallTraceGeneration(Params):

    # np.random.seed(seed=44864)
    # Params["Lags"] = np.random.choice([x for x in range(0,25000)],400).tolist()
      
    if __name__ == '__main__':
    
        t = time.time()

        
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        savedir = Misc.GetModelSavePath(ROOT_DIR,str(date.today()))
        DataSaveDir = os.path.join(ROOT_DIR, "Data")

        Misc.EmptyDataFolder(os.path.join( DataSaveDir,Params["Type"]))
        Misc.WriteDataParams( os.path.join( DataSaveDir,Params["Type"]),Params)


        AllCounts =[]
        Dt = DataConverter()
        Ds = DataLoader()
        
        
        
        for genome in Params["Genomes"]:

            SIMTRC     = SIMTraces.TSIMTraces(genome,Params["StretchingFactor"],0.34,0,Params["Enzyme"],Params["PixelSize"],Params['PixelShift'],Params[ "amplitude_variation"] ,Params["FPR"],Params["FPR2"],Params["FragmentSize"])  
            Gauss      = Misc.GetGauss1d(Params["FragmentSize"] , Misc.FWHMtoSigma(Misc.GetFWHM(Params["Wavelength"],Params["NA"],Params["ResEnhancement"])),Params["PixelSize"] )
            [Map,ReCutsInPx]  = SIMTRC.GetGenome(Params,genome)
            TraceGen   = TraceGenerator.TraceGenerator(SIMTRC, ReCutsInPx,Gauss,[],Ds, Dt,Params,DataSaveDir)
            TraceGen.SaveMap(Map)

            arg = [tuple([TraceGen,genome, t,Params]) for t in range(Params["NumTransformations"][Params["Genomes"].index(genome)]) ]
            pool = multiprocessing.Pool(processes=24)
            totcounts = pool.starmap(GenTraces, arg)
            pool.close()
            pool.join()   
            AllCounts= AllCounts+totcounts
            print('done')
            
        
            np.savez(os.path.join( DataSaveDir,Params["Type"],"NumberOfTraces.npz"),NumberOfTraces=np.sum(np.array(AllCounts)))
            print(str(time.time()-t) + " elapsed for generation" )

        
    
########### USER INPUT ############
    

Params = {"Wavelength" : 576,
               "NA" : 1.4,
               "FragmentSize" :300,
               "PixelSize" : 32.25*2,
               "ResEnhancement":1,
               "FromLags" :False,
               "Lags":[],
               "Enzyme" : 'TaqI',
               "NumTransformations"  :[10,10],
               "StretchingFactor" :[1.72],
               "LowerBoundEffLabelingRate" : 0.70,
               "UpperBoundEffLabelingRate" : 0.95,
               "amplitude_variation":[8.55696606597531,	3.23996722003733],
               "step" :2,
               "PixelShift": 0.2,
               "NoiseAmp": [5],
               "GenerateFullReference" :True,
               "LocalNormWindow":10000,
               "ZNorm": False,
               "Norm":  False,
               "Date" : str(date.today()),
               "Type" : "Training",
               "Genomes" : ['NC_000913.3'],
               "FPR": 0.4, #per kb 0.5
               "FPR2": 0.1, #per kb 0.2
               "Random-min": 52,
               "Random-max": 210}    


DataTypes = ["Training","Validation"]
# Enzymes   = ["TaqI","PabI"]
NumTransforms = [[400,400],[10,10]]


for i in range(0,2):
    # Params["Enzyme"] = Enzymes[i]
    Params["Type"]   = DataTypes[i]
    Params["NumTransformations"] = NumTransforms[i]
    CallTraceGeneration(Params)

   
###############################
    
    
    
    
    
    
