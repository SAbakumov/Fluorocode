# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 22:04:34 2020

@author: Boris
"""


import random,numpy as np
import time
import os
import csv
import matplotlib.pyplot as plt
from Core import Misc


class TraceGenerator():
    def __init__(self, SIMTRC, ReCutsInPx, Gauss, NoiseProfiles, DataStorage, DataHandler, Params, SaveDir):
        self.SimTraces = SIMTRC
        self.Gauss = Gauss
        self.ReCutsInPx = ReCutsInPx
        self.Noise = NoiseProfiles
        self.Ds = DataStorage
        self.Dt = DataHandler
        self.Params = Params
        for key, value in Params.items():
            setattr(self, key, value)

        self.ToAddLabeled = []
        self.ToAddRef = []
        self.ToAddLabels = []
        self.Positions = []
        if  SaveDir!=[]:
            if os.path.exists(SaveDir):
                self.SaveDir = SaveDir
            else:
                os.makedirs(SaveDir)
                self.SaveDir = SaveDir
            if not os.path.exists(os.path.join(self.SaveDir, self.Type)):
                os.makedirs(os.path.join(self.SaveDir, self.Type))   
        else:
            self.SaveDir = []
            
    def reset(self):
        self.ToAddLabeled = []
        self.ToAddRef = []
        self.ToAddLabels = []

                
                
     
        
        
    def ObtainTraces(self,batchnum,genome):
        t = time.time()
        
        EffLabeledTraces =[]
        ReferenceData   = []
        LabeledData     = []
        self.Ds.set_savingformat(self.SaveFormatAsCSV)
       
        for i in range(0,len(self.StretchingFactor)):
            self.SimTraces.set_stretch(self.StretchingFactor[i])
            self.SimTraces.set_recuts(self.ReCutsInPx[i],self.Gauss)
            # if self.FromLags:
            #     self.Lags = np.random.choice([x for x in range(0,np.int64(np.round(len(self.SimTraces.RefProfile)-self.FragmentSize-100)))],400).tolist()
            self.SimTraces.set_labellingrate(self.LowerBoundEffLabelingRate, self.UpperBoundEffLabelingRate)
            self.SimTraces.set_lags(self.FromLags,self.Lags,self.step)
            
            for offset in self.SimTraces.Lags:

                self.SimTraces.set_region(offset,self.FragmentSize,self.step)
                self.SimTraces.get_EffLabelledProfile()
                self.SimTraces.get_FPR()
                self.SimTraces.get_WrongRegions()
                trc = self.SimTraces.get_FluorocodeProfile(self.Gauss)[0]
                
                
                trc = np.squeeze(trc+self.NoiseAmp*np.random.uniform(0,1,self.FragmentSize))
                trcRef =50*self.SimTraces.RefProfile[self.SimTraces.region[0]:self.SimTraces.region[1]]

                self.ToAddLabeled.append( Misc.GetLocalNorm(trc,i,self.Params,self.SimTraces))
                self.ToAddRef.append( Misc.GetLocalNorm( trcRef,i,self.Params,self.SimTraces))
                self.ToAddLabels.append(self.ObtainLabel(genome))             
                self.Positions.append(self.SimTraces.region[0])

                
                
            EffLabeledTraces = EffLabeledTraces + self.ToAddLabeled
            ReferenceData  = ReferenceData + self.ToAddRef
            LabeledData    = LabeledData   + self.ToAddLabels
            
            self.reset()
        counts=self.Ds.BatchStoreData( EffLabeledTraces ,ReferenceData,LabeledData,self.Positions,self.Dt,self.Ds, os.path.join(self.SaveDir,self.Type) ,str(batchnum)+"-"+str(self.Genomes.index(genome)),self.Params)
        print('\n' + str(time.time()-t) ,end="")
        return counts , EffLabeledTraces
    
    
    def ObtainLabel(self, genome):
        if not hasattr(self, "Classes" ):
            lbl = np.zeros([len(self.Genomes)])
            lbl[self.Genomes.index(genome)] = 1
        else:
            lbl = np.zeros([len(self.Genomes)])
            lbl[self.Classes[self.Genomes.index(genome)]] = 1            
          
        return lbl
      
        
   
                
        
    def ObtainRandomTraces(self,maxNumDyes,minNumDyes, numprofiles,genome,batchnum):
      RandomTraces = []
      RandomLabels = []
      positions = []
      minNumDyesTotal = minNumDyes* Misc.PxTokb(self.FragmentSize, self.SimTraces)
      maxNumDyesTotal = maxNumDyes* Misc.PxTokb(self.FragmentSize, self.SimTraces)

      self.Ds.set_savingformat(self.SaveFormatAsCSV)

      for i in range(0,len(self.StretchingFactor)):
              
            for offset in range(0,numprofiles):

                numDyes =  np.random.randint(minNumDyesTotal, maxNumDyesTotal) 
                trace   =  np.zeros([self.FragmentSize])
                pos = np.random.uniform(0,self.FragmentSize,numDyes)
                u, c = np.unique(pos.astype(np.int16), return_counts=True)
                # c = c* np.random.gamma(self.amplitude_variation[0],self.amplitude_variation[1],size = c.shape)

                trace[u] = trace[u]+ c
                
                trace = self.SimTraces.GetFluorocodeProfile([trace],self.Gauss)[0]
                trc =  np.squeeze(trace+self.NoiseAmp*np.random.uniform(0,1,self.FragmentSize))

                RandomTraces.append( Misc.GetLocalNorm(trc,i,self.Params,self.SimTraces))
                RandomLabels.append( self.ObtainLabel(genome))

      
      counts = self.Ds.BatchStoreData(RandomTraces,[],RandomLabels,positions,self.Dt,self.Ds, os.path.join(self.SaveDir,self.Type),str(batchnum)+"-"+str(self.Genomes.index(genome)),self.Params)
      return counts, RandomTraces
      
    def SaveMap(self,Map,genome):
      with open(os.path.join(self.SaveDir,self.Type,'GEN-'+genome+'.csv'), 'w') as f: 
           write = csv.writer(f) 
           Map = [str(x) for x in Map]
           write.writerow(Map)
   

    def PlotNTraces(self, LabelledTraces,gene):   
        Traces = [item for labelledtraces in LabelledTraces for item in labelledtraces]
        plt.figure(figsize=(5,30)) 
        for i in range(1,10):
            plt.subplot(10, 1, i)
            plt.plot(random.choice(Traces))      

        plt.savefig(os.path.join(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode",gene+".png"))
                
    
