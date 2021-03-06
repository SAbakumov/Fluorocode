
# -*- coding: utf-8 -*-


"""
Created on Sun Sep 27 19:59:10 2020

@author: Sergey
"""
import Core.Misc as Misc
import Core.SIMTraces as SIMTraces
import Core.RandomTraceGenerator as R
import numpy as np
import json
import csv
import time
import os

from Core.DataHandler import DataConverter
from Core.DataHandler import DataLoader
from datetime import date
import multiprocessing
import Core.TraceGenerator as TraceGenerator
import csv 

def GenTraces(TraceGen,genome, transform,Params):
    # print('\n'+str(transform))
        
    if Params["GeneratorType"]=="FromFull" and genome!='Random':
        counts = TraceGen.ObtainTraces(transform,genome)
        
    elif genome == 'Random':
        counts = TraceGen.ObtainRandomTraces(Params["Random-max"],Params["Random-min"],40000,genome,transform)
    return counts

np.random.seed(seed=44864)
lags = np.random.choice([x for x in range(0,25000)],400).tolist()
            
if __name__ == '__main__':
    DL = DataLoader()
   
    t = time.time()
    Params = {"Wavelength" : 576,
            "NA" : 1.4,
            "FragmentSize" :300,
            "PixelSize" : 32.25*2,
            "ResEnhancement":1,
            "Lags": lags,
            "FromLags" : True,
            "NumTransformations"  :[1],
            "StretchingFactor" :[1.72],
            "LowerBoundEffLabelingRate" : 0.80,
            "UpperBoundEffLabelingRate" : 0.95,
            "amplitude_variation":[8.55696606597531,	3.23996722003733],
            "step" :200,
            "PixelShift": 0.2,
            "NoiseAmp": [5],
            "GenerateFullReference" :True,
            "LocalNormWindow":0,
            "ZNorm": False,
            "Norm":  False,
            "Date" : str(date.today()),
            "Type" : "Red",
            "Genomes" : ['NC_000913.3'],
            "FPR": 0.4, #per kb 0.5
            "FPR2": 0.1, #per kb 0.2
            # "Genomes" : ['NC_000913.3','NZ_LR735434.1','NZ_LR135297.1','NC_005139.1',4444446'NC_006177.1','NC_009663.1','NZ_CP021851.1'],
            "Random-min": 52,
            "Random-max": 210}
    
    
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    savedir = Misc.GetModelSavePath(ROOT_DIR,str(date.today()))
    
    json = json.dumps(Params)
    f = open(os.path.join(savedir ,'Params' + Params['Type'] +'.json'),"w")
    f.write(json)
    f.close()
    
    f = open(os.path.join(savedir ,'Params' + Params['Type']+'.csv'),"w")
    w = csv.writer(f)
    for key, val in Params.items():
        w.writerow([key, str(val)])        
    f.close()
    
    
    
    mypath =os.path.join( "D:\Sergey\FluorocodeMain\FluorocodeMain\Data",Params["Type"])
    for root, dirs, files in os.walk(mypath):
        for file in files:
            os.remove(os.path.join(root, file))

    
    

    AllCounts =[]
    Dt = DataConverter()
    Ds = DataLoader()
    
    
    
    for genome in Params["Genomes"]:
        
    

        SIMTRC     = SIMTraces.TSIMTraces(genome,Params["StretchingFactor"],0.34,0,'TaqI',Params["PixelSize"],Params['PixelShift'],Params[ "amplitude_variation"] ,Params["FPR"],Params["FPR2"],Params["FragmentSize"])  
        Gauss      = Misc.GetGauss1d(Params["FragmentSize"] , Misc.FWHMtoSigma(Misc.GetFWHM(Params["Wavelength"],Params["NA"],Params["ResEnhancement"])),Params["PixelSize"] )
        [Map,ReCutsInPx]  = SIMTRC.GetGenome(Params,genome)
        TraceGen   = TraceGenerator.TraceGenerator(SIMTRC, ReCutsInPx,Gauss,[],Ds, Dt,Params)

        if Params["NumTransformations"]!='FromExisting':
            arg = [tuple([TraceGen,genome, t,Params]) for t in range(Params["NumTransformations"][Params["Genomes"].index(genome)]) ]
            pool = multiprocessing.Pool(processes=24)
            totcounts = pool.starmap(GenTraces, arg)
            pool.close()
            pool.join()   
            AllCounts= AllCounts+totcounts
            print('done')
            
   
    try:       
        np.savez(os.path.join(mypath,"NumberOfTraces.npz"),NumberOfTraces=np.sum(np.array(AllCounts)))
        print(str(time.time()-t) + " elapsed for generation" )
    except:
        print('Done')
    
  
    
    
        
        
    

    
    
    
    
    
    
    
    
    
    
