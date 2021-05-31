# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 19:20:34 2020

@author: Sergey
"""

import numpy as np
import random


# class RandomTraceGenerator:
#     def __init__(self,avlength,sigmalength,numsamples) :
#         self.avlength = avlength
#         self.sigmalength = sigmalength
#         self.numsamples = numsamples
        
        
def stratsample(arr,avlength, sigmalength, numsamples):
    
    AllTraces  = [];
    for i in range(0, int(round(max(arr)-round(3*avlength))),round(3*avlength)):
        FirstInd = np.asscalar(np.argwhere(arr>=i)[0])
        LastInd  = np.asscalar(np.argwhere(arr>=i+round(3*avlength))[0])
        SubArray = arr[FirstInd:LastInd]
        for j in range(0,round(numsamples/(max(arr)/(3*avlength))).astype(int)):
            StartIndexOfTrace = np.random.choice(SubArray)
            EndIndexOfTrace = StartIndexOfTrace + np.random.normal(loc = avlength,scale = sigmalength)
            
            FirstInd =np.asscalar( np.argwhere(arr>=StartIndexOfTrace)[0])
            try:
                LastInd  =np.asscalar( np.argwhere(arr>=EndIndexOfTrace)[0])
            except:
                LastInd = len(arr)-1

    
            SubTrace = arr[FirstInd:LastInd]
            AllTraces.append(SubTrace)
            
            
 
    return AllTraces

def GetEffLabelingRate(Traces,LabelingEfficiency):
    EffLabeledTraces = []
    for trc in Traces:
        EffLabeledTraces.append(np.random.choice(trc,np.int(LabelingEfficiency*len(trc)),replace=False))

    return EffLabeledTraces

    
        # if LabelingEfficiency!=1:
        #     LabelTransform = lambda x : np.sort(np.array(random.sample(list(x),int(LabelingEfficiency*len(x)))))
        #     EffLabeledTraces = list(map(LabelTransform, Traces))
            
        # else:
        #     EffLabeledTraces =[np.sort(np.squeeze(np.array(Traces)))]
        # return EffLabeledTraces
        
def GetCombinedRandomizedTraces(ListOfRandomTraces,numclasses):
    TotalCombinedList = []
    return TotalCombinedList
    
    

            
def GetRandomFixedLengthTraces(arr,length,numsamples):
  AllTraces  = []
  
  # step = np.max(arr)/self.numsamples
  step = np.max(arr)/numsamples

  for pos in range(0,int(np.round( np.max(arr))),int(step)):
       StartIndexOfTrace = pos
       
       # StartIndexOfTrace = np.random.choice(arr)
       EndIndexOfTrace = StartIndexOfTrace + length
       
       FirstInd =np.asscalar( np.argwhere(arr>=StartIndexOfTrace)[0])
       try:
           LastInd  =np.asscalar( np.argwhere(arr>=EndIndexOfTrace)[0])
       except:
           continue

      
       SubTrace = arr[FirstInd:LastInd]
       AllTraces.append(SubTrace)
          
          
   
  return AllTraces    


        
            


        