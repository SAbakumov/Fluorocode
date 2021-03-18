# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 12:37:06 2020

@author: Sergey
"""
import numpy as np
from Bio import Entrez
from Bio import SeqIO
import Core.Misc as msc
import Core.RandomTraceGenerator as RTG
import random
import time
import os

class TSIMTraces:
      def __init__(self, Species, Stretch, BPSize, Optics,Enzyme,PixelSZ,Shift,ampvar,FPR,FPR2,frag_size):
        self.Species = Species
        self.Stretch = Stretch
        self.BPSize = BPSize
        self.Optics = Optics
        self.Enzyme = Enzyme
        self.PixelSize = PixelSZ
        self.PixelShift = Shift
        self.AmplitudeVariation = ampvar
        self.FPR = FPR
        self.FPR2 = FPR2
        self.Trace = []
        self.RandomTraces = []
        self.Map = []
        self.frag_size = frag_size
        
      def set_stretch(self, stretch):
          self.Stretch = stretch
        
      def set_recuts(self,recuts,gauss):
          self.recuts = recuts
          FullTrace = self.GetFullProfileIdeal(RTG.GetEffLabelingRate([self.recuts] ,1))
          self.RefProfile= self.GetFluorocodeProfile(FullTrace,gauss)[0]
      
      def set_region(self, *args):
          shft = np.random.randint(0,args[2])
          self.region = [args[0]+shft,args[0]+shft+args[1]]
          
      def set_lags(self,FromLags,Lags,step):
          if FromLags:
              self.Lags = Lags
          else:
              self.Lags = list( range(20,len(self.RefProfile)-self.frag_size-20,step))
                 
      def set_labellingrate(self,Up,Low):
          self.Up = Up
          self.Low = Low
        
      def get_labellingrate(self):
          labelrate = np.random.uniform(self.Low,self.Up)
          return labelrate
      
      def get_EffLabelledProfile(self):
          self.FullTrace = self.GetFullProfile(RTG.GetEffLabelingRate([self.recuts] ,self.get_labellingrate()))
        
      def get_FPR(self):
          self.FullTrace = self.YieldFPR(self.FullTrace)
          
      def get_WrongRegions(self):
          self.FullTrace = self.YieldWrongRegions(self.FullTrace)
      def get_FluorocodeProfile(self,gauss):
          trc = self.FullTrace[self.region[0]:self.region[1]]
          conv_trace = self.GetFluorocodeProfile([trc],gauss)
          return conv_trace
        
          
      
        
      
        
      def GetTraceRestrictions(self):
        Entrez.email = "abakumov.sergey1997@gmail.com"
        
        
        ROOT_DIR = os.path.abspath(os.curdir)
        DataBasePath = os.path.join(ROOT_DIR, 'DataBases')
        search_term = self.Species
        
        FileName =  '%s.fasta' % search_term
        
        if not os.path.exists(os.path.join(DataBasePath,FileName )):
            handle = Entrez.efetch(db="nucleotide", id=search_term, rettype="fasta", retmode= 'text')
            
            f = open(os.path.join(DataBasePath,FileName ), 'w')
            f.write(handle.read())
            f.close()
        
        genome = SeqIO.parse(os.path.join(DataBasePath,FileName ), "fasta")


        for record in genome:
            CompleteSequence = record.seq
            
        cuts = msc.rebasecuts(self.Enzyme,CompleteSequence )
        
        
        return cuts
    
    
    
    
    
      def GetDyeLocationsInPixel(self,ReCuts,strtch):
        ReCuts = np.array(ReCuts)
        # ReCuts = ReCuts-ReCuts[0]        
        ReCutsInPx = msc.kbToPx(ReCuts,[strtch, self.BPSize,self.PixelSize])
          
        return ReCutsInPx
    
      def GetTraceProfile(self,trace,gauss,size,orrarr):
        x = np.zeros(size)
        trace = trace-np.min(trace)
        for i in range(0,len(trace)):
            try:
                x[int(np.round(trace.item(i)))] = x[int(np.round(trace.item(i)))]+1+random.uniform(-0.2,0.2)
            except:
                continue
            
        signal = np.convolve(x,gauss, mode = 'same')
        signal = msc.ZScoreTransform(signal)
        return signal
    
      def GetFluorocodeProfile(self,trace,gauss):
        signals = []
        for trc in trace:
            signals.append(np.convolve(trc,gauss, mode = 'same'))
        return signals
    
    

        # return trace
    
      def GetFullProfile(self,genome):
        genome = genome[0]

        genome =(genome+ np.random.uniform(-self.PixelShift,self.PixelShift,size = genome.shape)).astype(int)
        Trace = np.zeros((np.max(genome)+10).item())
        u, c = np.unique(genome, return_counts=True)
        # c =  c*np.random.gamma(self.AmplitudeVariation[0],self.AmplitudeVariation[1],size = c.shape)
        # c =  c+c*np.random.uniform(-0.2,0.2,size = c.shape)
        c = self.GetDyeAmp(c,-0.2,0.2)
        Trace[u]=Trace[u]+ c
        return Trace
        
      def GetDyeAmp(self,c,AmpVarMin,AmpVarMax):
          # c =  c+c*np.random.uniform(-0.2,0.2,size = c.shape)
          c = c* np.random.gamma(self.AmplitudeVariation[0],self.AmplitudeVariation[1],size =c.shape)
          return c

      def YieldFPR(self,trc):
        if self.FPR>0:
            num_dyes = int( msc.PxTokb(self.frag_size, self)*self.FPR/1000)
            fpr_locs = np.random.uniform(self.region[0],self.region[1],num_dyes )
            fpr_amps = self.GetDyeAmp(np.ones(fpr_locs.shape),-0.2,0.2)
            # fpr_amps = np.random.gamma(self.AmplitudeVariation[0],self.AmplitudeVariation[1],size = num_dyes)
            trc[(fpr_locs).astype(np.int64)] =  trc[(fpr_locs).astype(np.int64)]+fpr_amps

            if self.FPR2>0:
                num_dyes2 = int( msc.PxTokb(self.frag_size, self)*self.FPR2/1000)
                fpr_locs = np.random.uniform(self.region[0],self.region[1],num_dyes2)
                # fpr_amps = 2*np.random.gamma(self.AmplitudeVariation[0],self.AmplitudeVariation[1],size = num_dyes2)
                fpr_amps = 2*self.GetDyeAmp(np.ones(fpr_locs.shape),-0.2,0.2)

                trc[(fpr_locs).astype(np.int64)] =  trc[(fpr_locs).astype(np.int64)]+fpr_amps
    

        return trc
      def YieldWrongRegions(self,trc):
          numregs = np.random.randint(0,3)
          for i in range(numregs):
              startind = self.region[0]+ np.random.randint(0,self.frag_size)
              endind = startind + np.random.randint(25,35)
              trc[startind:endind] = np.random.permutation(trc[startind:endind])
          return trc
              
      def YieldNonLinearStretch(self,trc,region,frag_size):
              
          startind = region
          endind = startind + frag_size
          
          numzeros = np.random.randint(0,5)
          regind = np.random.randint(0,endind-numzeros, numzeros)

          arr = trc[startind:endind]
          arr = np.insert(arr,regind,0)
          
          trc[startind:endind] = arr[0:frag_size]
          return trc
                            
          
          
          
      def GetFullProfileIdeal(self,genome):
        # genome = genome[0]
        Traces = []
        for gen in genome:
            Trace = np.zeros([int(np.round(np.max(gen)).item())])
            for i in range(0,len(gen)):
                try:
                    # pos = int(np.round(genome[i].item()+np.random.uniform(-1.5,1.5)))
                    pos = int(np.round(gen[i].item()))
    
                    # Trace[pos] =  Trace[pos]+1+np.random.uniform(-0.1,0.1)
                    Trace[pos] =  Trace[pos]+1
                except:
                    # pos = int(np.round(genome[i].item()+np.random.uniform(-2,2)))
    
                    pos = int(np.round(gen[i].item()))
            
            Traces.append(Trace)
        return Traces
    
    
      def GetGenome(self,Params,genome):
        ReCuts = []
        if not 'Artificial' in genome and not 'Random' in genome:
            
            ReCuts     = self.GetTraceRestrictions()
            ReCutsInPx = []
            for strtch in Params["StretchingFactor"]:
                ReCutsInPx.append(self.GetDyeLocationsInPixel(ReCuts,strtch))
        elif 'Artificial' in genome:
            try:
                ReCutsInPx = np.empty(Params["ArtificialDyeNumber"])
                for i in range(0,Params["ArtificialDyeNumber"]):
                    ReCutsInPx[i] = random.uniform(0,Params["ArtificialGenomeLen"]  ) 
            except:
                print( "AritificialDyeNumber" in locals() or  "ArtificialGenomeLen" in locals(), "The length or number of dyes of artificial genome is not specified." ) 
        elif 'Random' in genome:
            ReCutsInPx = []
            
        return ReCuts ,ReCutsInPx  


