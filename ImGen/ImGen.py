# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 12:59:13 2020

@author: Sergey
"""

# import tensorflow as tf
# import matplotlib as plt
# from tensorflow import keras
# from tensorflow.keras import layers
import numpy as np
import os
from scipy.signal import convolve2d
from Core.SIMTraces import TSIMTraces
import Core.Misc as msc
import random
from PIL import Image
from scipy.ndimage.interpolation import rotate


class TrainImageGenerator(TSIMTraces):
    def __init__(self, TrainDir,NumImages,NumTracesPerImage,Resolution,Wavelength,NA,pixelsize):
        if not(os.path.exists(TrainDir)):            
            os.mkdir(TrainDir)   
        self.TrainDir = TrainDir
        self.NumImages = NumImages
        self.NumTracesPerImage = NumTracesPerImage
        self.Resolution = Resolution
        self.Wavelength = Wavelength
        self.NA         = NA
        self.pixelsize  = pixelsize
        self.numclasses = 1
        
        
        self.Images = []
        self.Labels = []
        
    # @classmethod
    # def from_super_instance(cls, TrainDir,NumImages,NumTracesPerImage,Resolution,Wavelength,NA, super_instance):
    #     return cls(TrainDir,NumImages,NumTracesPerImage,Resolution,Wavelength,NA, **super_instance.__dict__)

        
    def Generate(self, LabeledTraces, numclass):
        AllImages = []
        AllLabels = []
        TraceNum = 0
        for image in range(0,self.NumImages):
            img = np.float32(np.zeros([self.Resolution,self.Resolution]))
            labelimg = np.zeros([self.Resolution,self.Resolution],dtype=int)
            GetInitialYPos = 0
            for tracenum in range(TraceNum,TraceNum+self.NumTracesPerImage):
                if TraceNum+self.NumTracesPerImage>len(LabeledTraces):
                    TraceNum =0
                TraceNum+=1
                trace = LabeledTraces[TraceNum]
                trace = trace-trace[0]
                
                GetInitialXPos = np.random.randint(-trace.item(len(trace)-1),self.Resolution-50)
                
                step = round(self.Resolution/self.NumTracesPerImage)
                if step < 16:
                    step = 16
                    
                
                GetInitialYPos = np.random.randint(GetInitialYPos+step-6,GetInitialYPos+step+15)
                if GetInitialYPos > self.Resolution-(step+15):
                    break
                
                
                trace = trace+GetInitialXPos
                InitIndex = np.argwhere(trace>0)
                FinIndex  = np.argwhere(trace<self.Resolution)
                
                InitIndex = InitIndex.item(0)  
                FinIndex = FinIndex.item(len(FinIndex)-1)-1
                   
                    

                
                Positions  = trace[InitIndex:FinIndex]
                # CorrectedPositions = [x for x in Positions if x>0 and x<self.Resolution]
                
                if len(Positions)!=0:
                    labelimg = self.GetLabel(labelimg,[Positions[0],Positions[-1]],GetInitialYPos, np.int32(numclass))
                    # return labelimg
                    for index in Positions:
                        if round(index)<self.Resolution:
                            img[int(round(index))][GetInitialYPos] = img[int(round(index))][GetInitialYPos]+1
                        
            img = self.GetImage(img)
            print('\r'+ str(image) + " from " + str(self.NumImages) + " done", end = ""  )
            AllImages.append(img)
            AllLabels.append(labelimg)
            
            
            self.Images = AllImages
            self.Labels = AllLabels
            
        return AllImages,AllLabels    
                
        # return img
    
    def GetLabel(self,lbimg, Xpos,Ypos,numclass):
        xin = Xpos[0].astype(int)
        xlast = Xpos[1].astype(int)


        lbimg[xin:xlast,Ypos-1:Ypos+4]=numclass
        return lbimg
        
    
    def GetImage(self, img):
        FWHM = msc.GetFWHM(self.Wavelength,self.NA)
        sigma = msc.FWHMtoSigma(FWHM) 
        GaussToConvolve = np.float32(msc.GetGauss(sigma, self.pixelsize))
        
        Image = convolve2d(img,GaussToConvolve,mode='same')
        return Image
        
        
        
    def ImAugment(self,numaugmentations):
        AugmentedImages = []
        AugmentedLabels = []
        for image in range(0,len(self.Images)):
            AugmentedImages.append(self.Images[image])
            AugmentedLabels.append(self.Labels[image])
            for i in range(0,numaugmentations):
                im = self.Images[image]
                lb = self.Labels[image]
                
                
                angle = random.uniform(-5, 5)
                im = rotate(im, angle,reshape = False)
                lb = rotate(lb, angle,reshape = False)
                AugmentedImages.append(im)
                AugmentedLabels.append(lb)

        return AugmentedImages, AugmentedLabels
            
        
        
    
            
    
        
        
# Images =  TrainImageGenerator('D:\Sergey\TrainDirectory', 1, 50, 512,Wavelength =510 , NA = 1.4 )


# class LabelImageGenerator:
#     def __init__(self, TrainDir):
