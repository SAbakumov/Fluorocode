# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 19:55:53 2020

@author: Sergey
"""

from Bio import Restriction
import Core.SIMTraces
import numpy as np
import sys
import os

from datetime import date
from scipy import sparse
from scipy.sparse.linalg import spsolve


def baseline_als(y, lam, p, niter=10):
  L = len(y)
  D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z

def ReformatListArrayToSize(array,strch,CorrectNum, sz):
    TotArray = []
    Strch = []
    Num = []
    for  i in range(0,len(array)):
        # for i in range(0,len(item)-sz):
        try:
            TotArray.append(np.reshape(array[i][0:256],[256,1]))
            Strch.append(strch[i])
            Num.append(CorrectNum[i])
        except:
            print("Too small")
    return TotArray,Strch,Num

def GetGauss1d(size,sigma,pixelsz):
    x = np.linspace(-np.round(size/2),np.round(size/2), size+1)*pixelsz
    Gauss = np.exp(-np.power(x,2)/(2*np.power(sigma,2)))
    Gauss = Gauss[0:len(Gauss)-1]
    return Gauss

# def GetPerlinNoise(size):
#     noise=perlin.Perlin(50)

#     time=[i for i in range(size)]
#     values=np.array([noise.valueAt(i) for i in time])
#     return values


def GetGauss(sigma,pixelsz):
    x = np.linspace(-18,17)*pixelsz
    y = np.linspace(-18,17)*pixelsz
    
    xv,yv = np.meshgrid(x,y)
    Gauss = np.multiply(np.exp(-np.power(xv,2)/(2*np.power(sigma,2))),np.exp(-np.power(yv,2)/(2*np.power(sigma,2))))
    return Gauss
    
def GetFilename(File):
    
    Filename = File.rpartition('\\')[-1]
    return Filename
    
def update_progress(progress):
   barLength = 1 # Modify this to change the length of the progress bar
   status = ""
   if isinstance(progress, int):
       progress = float(progress)
   if not isinstance(progress, float):
       progress = 0
       status = "error: progress var must be float\r\n"
   if progress < 0:
       progress = 0
       status = "Halt...\r\n"
   if progress >= 1:
       progress = 1
       status = "Done...\r\n"
   block = int(round(barLength*progress))
   text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
   sys.stdout.write(text)
   sys.stdout.flush()
   

def GetFWHM(wavelength,NA,ResEnhancement):
    FWHM =  0.61*wavelength/(NA)/ResEnhancement
    return FWHM
    
def FWHMtoSigma(FWHM):
    sigma = FWHM/2.3548
    return sigma 

def SigmatoFWHM(Sigma):
    FWHM = Sigma*2.3548
    return FWHM
    
def rebasecuts(Enzyme, Strand):
    batch = Restriction.RestrictionBatch()
    batch.add(Enzyme)
    enzyme = batch.get(Enzyme)    
    
    Sites = enzyme.search(Strand)
    
    return Sites

def GetArray(X_Data):
    X_DataPast = np.reshape(X_Data[:,0:256,0],[X_Data.shape[0],256,1])
    # X_DataFuture = np.reshape(X_Data[:,256:256+32,0],[X_Data.shape[0],32,1])
    X_DataFuture = 0
    X_DataPast = np.reshape(X_DataPast,[X_DataPast.shape[0],4,64,1])

    return X_DataPast, X_DataFuture


def kbToPx(arr,args):
    if type(args)==list:
        stretch, nmbp, pixelsz = args[0] , args[1] ,  args[2]
    else:
        stretch, nmbp, pixelsz = args.Stretch , args.BPSize ,  args.PixelSize

        
    arr  =   (arr*stretch*nmbp)/pixelsz
    return arr

def PxTokb(arr,args):
    if type(args)==list:
        stretch, nmbp, pixelsz = args[0] , args[1] ,  args[2]
    elif type(args)==Core.SIMTraces.TSIMTraces:
        stretch, nmbp, pixelsz = args.Stretch , args.BPSize ,  args.PixelSize
    else:
        print('Unsupported data type in kbToPx, aborting execution')
    
    
    arr  =   (arr/stretch/nmbp)*pixelsz
    return arr
    
def ZScoreTransform(trace):
    trace = (trace - np.mean(trace))/np.std(trace)
    return trace

def GetModelSavePath(ROOT_DIR,date):
    path  = os.path.join(ROOT_DIR , 'StoredModels')
    if os.path.exists(os.path.join(path,date)):
        dr = os.path.join(path,date)
        dirs = os.listdir(os.path.join(path,date))   
    else:
        dr = os.path.join(path,date)
        os.makedirs(os.path.join(path,date))
        dirs = os.listdir(os.path.join(path,date))   


    if len(dirs)==0:
        os.makedirs(os.path.join(dr, 'Training_1'))
    dirs = os.listdir(os.path.join(path,date))   
   
        
    numericals = []
    for directory in dirs:
       charts = list(directory)
       numericals = numericals + [int(s) for s in charts if s.isdigit()]
   
    maxval = max(numericals)
    if len(os.listdir(os.path.join(dr, 'Training_' + str(maxval))))>7:
        os.makedirs(os.path.join(dr, 'Training_' + str(maxval+1)))
        savedir =  os.path.join(dr, 'Training_' + str(maxval+1))
    else:
        savedir = os.path.join(dr,'Training_'+str(maxval))
    return savedir

           
def GetLocalNorm(trace,i,Params,SimTraces):
    if Params["LocalNormWindow"]>0:
        conv = [Params["StretchingFactor"][i] , SimTraces.BPSize ,  SimTraces.PixelSize]

        trace  = np.round(normalize_local(kbToPx(Params["LocalNormWindow"],conv),trace)*100).astype(np.int16)
    elif Params["ZNorm"]:
        trace = (((trace- np.mean(trace))/np.std(trace))*100).astype(np.int16)
    elif Params["Norm"]:
        trace = (trace/np.std(trace)*100).astype(np.int16)
    else:
        trace = (trace).astype(np.int16)
     
    return trace          
                       

def GetLocalNormFromPars(trace,strch, bpsize,pixelsize,window):
    conv = [strch , bpsize ,  pixelsize]

    trace  = normalize_local(kbToPx(window,conv),trace)*100

    return trace          
                      
    
def normalize_local(npoints, trace):
  npoints = int(npoints)
  window = np.ones([npoints,1], dtype = int)/npoints
  trace2 = np.reshape(trace, trace.size)
  window = np.reshape(window,window.size)
  local_mean = np.convolve(trace2,window,'same')
  out = np.subtract(trace2 ,local_mean)
  out2 = out*out
  local_var = np.convolve(out2,window,'same')
  if local_var[local_var > 0].sum() > 0:
     local_var[local_var == 0] = min(local_var[local_var > 0])
     out = out/np.sqrt(local_var);
  return out
        
        
    
        
    
    
