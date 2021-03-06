# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 14:23:08 2020

@author: Sergey
"""

from Bio import Restriction
import Core.SIMTraces
import numpy as np
import os
import tifffile as tiff
import cv2
import Core.Misc as msc
# import tensorflow as tf
import scipy
import scipy.io
import sklearn.utils
import random
import sys
import csv

class DataLoader():
    def __init__(self):
        self.TrainingImages = []
        self.LabelImages = []
    
    def PrepareTrainingData(self,folder):
        print('Loading Training images from folder ' + folder)
        TrainingImages = []
        Files = os.listdir(folder)
        prog = 0
        for file in Files:
            prog = prog+1
            img = DataLoader.ReturnImage(os.path.join(folder,file))
            TrainingImages.append((msc.GetFilename(file), np.reshape( img,[img.shape[0],img.shape[1],1]))) 
            if prog % 500 == 0:
                print(str(prog) + ' out of ' + str(len(Files))+ ' done')
                
                
        self.TrainingImages = TrainingImages
        return TrainingImages
            
    def ReturnImage(directory):
        img= cv2.imread(directory,-1)

        return img
    def LoadTracesFromCSV(self,path):
        TotalMatches = []
        with open(path, newline='') as csvfile:
            DNALines = csv.reader(csvfile, delimiter=',')
            AllDNALines = []
            for DNALine in DNALines:
                AllDNALines.append(DNALine)
            for i in range(0,len(AllDNALines)):
                TotalMatches.append(np.array([float(x) for x  in AllDNALines[i] ]))

        return TotalMatches

    
    def LoadMatchedTracesFromCSV(self,path,respath, size):
        with open(path, newline='') as csvfile:
            DNALines = csv.reader(csvfile, delimiter=',')
            AllDNALines = []
            for DNALine in DNALines:
                AllDNALines.append(DNALine)
           
        with open(respath, newline='') as csvfile:
            MatchLines = csv.reader(csvfile, delimiter=',')
            Matches = []
            for match in MatchLines:
              try:
                  Matches.append([match[2],match[6],float(match[10]),float(match[5]),match[0],match[8]])
              except:
                  Matches.append([])
                  continue
        Matches.pop(0)
        TotalMatches = []
        Lags = []
        Strch = []
        Direc = []
        CorrectNum = []
        Strch = []
        Pval = []
        for i in range(0,len(Matches)):
    
            try:
                # if float(Matches[i][0])==1:
                   if Matches[i][1]=='default':
                       TotalMatches.append(np.array([float(x) for x  in AllDNALines[i]]))
                       Lags.append(Matches[i][2])
                       Strch.append(Matches[i][3])
                       Direc.append(Matches[i][1])
                       CorrectNum.append(Matches[i][4])
                       Pval.append(Matches[i][5])
                   elif Matches[i][1]=='flipped':
                       
                       TotalMatches.append(np.squeeze(np.flipud(np.reshape(np.array([float(x) for x  in AllDNALines[i]]),[len( AllDNALines[i]),1]))))
                       Lags.append(Matches[i][2])
                       Strch.append(Matches[i][3])
                       Direc.append(Matches[i][1])
                       CorrectNum.append(Matches[i][4])
                       Pval.append(Matches[i][5])



            except:
               continue
       
        return TotalMatches,Lags,Strch,Direc, CorrectNum,Pval
                
    def LoadPValsFromCSV(self,path,respath, size,true_genomes):
        with open(path, newline='') as csvfile:
            DNALines = csv.reader(csvfile, delimiter=',')
            AllDNALines = []
            for DNALine in DNALines:
                AllDNALines.append(DNALine)
           
        with open(respath, newline='') as csvfile:
            MatchLines = csv.reader(csvfile, delimiter=',')
            matchFields =[x for x in MatchLines]
        to_load_genome = []

        PValTrue = []
        PValFalse = []


        TotalMatches = []
        fields =matchFields[0]
        for field in fields:
            # for num_field in range(0, len(field)):
            if "reference species" in field:
                to_load_genome.append(fields.index(field))
                    
        for i in range(1,len(matchFields)):
            PValMinTrue = []
            PValMinFalse = []          
            for num_field in to_load_genome:
   
                if any([genome in matchFields[i][num_field] for genome in true_genomes]):
                    PValMinTrue.append(float(matchFields[i][num_field+5]))
                else:
                    PValMinFalse.append(float(matchFields[i][num_field+5]))
                
                
            PValMinTrue = np.array(PValMinTrue)
            PValMinFalse = np.array(PValMinFalse)
            
            indMinTrue = np.argmin(PValMinTrue)
            indMinFalse = np.argmin(PValMinFalse)
            TotalMatches.append(np.squeeze(np.flipud(np.reshape(np.array([float(x) for x  in AllDNALines[i]]),[len( AllDNALines[i]),1]))))
            PValTrue.append(PValMinTrue[indMinTrue])
            PValFalse.append(PValMinFalse[indMinFalse])
            
                    # if "reference species" in to_load_genome
                
                    
                    
            
       
        return  PValFalse, PValTrue        
    def PrepareLabeledData(self, folder):
        print('Loading corresponding label images from folder ' + folder)
        LabelImages = []
        Files = os.listdir(folder)
        prog = 0

        
        
        for TrainImage in self.TrainingImages:
            Item = [value for value in Files  if  TrainImage[0] in value]
            prog = prog+1
            img =DataLoader.ReturnImage(os.path.join(folder,Item[0]))
            img = np.sum(img,axis=2)
            img[~(img==765)] = 1
            img[(img==765)] = 0   
            LabelImages.append((Item, np.reshape(img,[img.shape[0],img.shape[1],1]))) 
            if prog % 500 == 0:
                print(str(prog) + ' out of ' + str(len(Files))+ ' done')
            
        self.LabelImages = LabelImages
        return LabelImages
        
    def LoadFromMatlab(self,path,file):
        data= scipy.io.loadmat(os.path.join(path,file))
        
        trc =data['Traces']
        C = np.squeeze(trc)
        Traces = []
        for i in range(C.shape[0]):
            Traces.append(np.squeeze(C[i]))
            
        Stretch = data['Strtch']
        Lags = data['Lags']
        
        return Traces, Stretch, Lags
    
    def BatchLoadTrainingData(self,path):
        
        files = os.listdir(path)
        maxind = len(files)
        # maxind = 20
        counts = np.load(os.path.join(path,"NumberOfTraces.npz"))["NumberOfTraces"]

        for i in range(0,maxind):
            sys.stdout.write("\r" + str(i))
            sys.stdout.flush()
            if files[i]!="NumberOfTraces.npz" and ".npz" in files[i] :
                    if i==0:
                    
                        Train ,Labels, Ref,pos = self.LoadTrainingData(os.path.join(path,files[i]))
                        # stadDev = np.std(Train)
                        assignmentarr  =  (np.linspace(0,counts,counts)).astype(np.int64)
                        np.random.shuffle(  assignmentarr)
                        PreTrain = np.zeros([counts+maxind-1,Train.shape[1],1],dtype=np.int16)
                        PreRefs = np.zeros([counts+maxind-1,Train.shape[1],1],dtype=np.int16)
                        PrePos  = np.zeros([counts+maxind-1])
                        PreLabels = np.zeros([counts+maxind-1,Labels.shape[1],1])
    
                        for j in range(0,Train.shape[0]):
                          
                            PreTrain[assignmentarr[j],:,:] = Train[j,:,:]
                            if len(Ref)!=0:

                                PreRefs[assignmentarr[j],:,:] = Ref[j,:,:]
                                PrePos[assignmentarr[j]]  = pos[j]
                            PreLabels[assignmentarr[j],:,:] = Labels[j,:,:]
                            endcounts = j+1
                    else:
                        TrainImages, LabelImages,Refs ,pos= self.LoadTrainingData(os.path.join(path,files[i]))

                        allcounts = endcounts
                        for j in range(endcounts,endcounts+TrainImages.shape[0]):
 
                            PreTrain[assignmentarr[j],:,:] =( TrainImages[j-allcounts,:,:]).astype(np.int16)
                            if len(Refs)!=0:
                                PreRefs[assignmentarr[j],:,:] =( Refs[j-allcounts,:,:]).astype(np.int16)
                                PrePos[assignmentarr[j]]  = pos[j-allcounts]
                            PreLabels[assignmentarr[j],:,:] = LabelImages[j-allcounts,:,:]
                            endcounts = j+1

    
        return PreTrain, PreRefs, PreLabels, PrePos
    
    def LoadTrainingData(self,path):
        
        Data = np.load(path,allow_pickle=True )
        a = Data.files
        # TrainImages = (Data['training_data'].astype(np.float32)/pow(2,16)).astype(np.float32)
        try:
            TrainImages = Data['training_data'].astype(np.float32)
            ReferenceImages = Data['training_refs'].astype(np.float32)
            LabelImages = Data['training_labels'].astype(np.float32)
            pos = Data['pos'].astype(np.float32)

        except:
            TrainImages = Data['training_data']
            ReferenceImages = Data['training_refs']
            LabelImages = Data['training_labels']
        return TrainImages, LabelImages,ReferenceImages,pos
    
    def SaveTrainingData(self,training_data,training_refs, training_labels,positions, path):
        np.savez_compressed(path,training_data=training_data,training_refs=training_refs,training_labels= training_labels,pos = positions)
       
        
     
    def BatchStoreData(self,EffLabeledTraces,ReferenceData,LabelData,Positions, Dt,Ds,path,batchnum,Params):
        AllLabels = []
        AllProfiles = []
        AllRefs = []
        Profiles = []
        Refs = []
        Labels = []
        progress = 0
        # counts = np.load(os.path.join(path,"NumberOfTraces.npz"))["NumberOfTraces"]
        for x in EffLabeledTraces:  
            Profiles.append(x) 
            # progress+=1
            # print('\r'+ str(progress) + " from " + str(len(EffLabeledTraces)) + " done", end = ""  )
        
        
        for x in ReferenceData:  
            Refs.append(x)
            
        for x in LabelData:
            Labels.append(x)
            
            
        AllRefs.append(Refs)
        AllProfiles.append(Profiles)
        AllLabels.append(Labels)


            
        
        
        Profiles= Dt.ToNPZ1D(AllProfiles,2,datatype='data')
        Refs= Dt.ToNPZ1D(AllRefs,2,datatype='data')
        Labels= Dt.ToNPZ1D(AllLabels,2,datatype='data')
        counts = len(Profiles)
        
        # np.savez(os.path.join(path,"NumberOfTraces.npz"),NumberOfTraces=counts)
        # print('\r'+"Saving...")
        Ds.SaveTrainingData(Profiles,Refs,Labels,Positions ,path=path+ "\DataFor" +Params["Type"]+ batchnum + ".npz")      
        return counts
        
    def PrepMatlabData(self,path,file,sz):
        prfs, strch, lags = self.LoadFromMatlab(path,file)

        data = []
        stretch= []
        lag = []
        for ind in range(0, len(prfs)):
            profile = prfs[ind]
            for i in range(0,len(profile)-sz):
                data.append(profile[i:i+sz])
                stretch.append(strch[ind])
                lag.append(lags[ind]+i)
        return np.array(data) ,  np.array(stretch), np.array(lag)
                
            
        
        
       
class DataConverter():
    
    # def ToTensor(self,array):
    #     for item in  array:
    #         item  = tf.convert_to_tensor(item)
    #     return array
    
    
    # def ToOneHot(self,img,numclass,numclasses):
    #     image  = tf.one_hot(img,numclasses)
    #     return image.numpy()
        
    def ToNPZ(self,imgArrays,numclasses):
        DataTensor = np.empty(tuple([len(imgArrays[0])*2+1]) +np.shape(imgArrays[0][0])).astype(np.int16)
        ShuffledArrays = []
        for numclass in range(0,numclasses):
            ClassArray = imgArrays[numclass]
            random.Random(452856).shuffle(ClassArray)
            ShuffledArrays.append(ClassArray)
            
        imgArrays = ShuffledArrays

        if len(np.shape(imgArrays[0][0]))!=3:     
            DataTensor = np.expand_dims(DataTensor,axis = 3)
        nextimg = 0
        for image in range(0, len(imgArrays[0])):
            print(image)
            for numclass in range(0,numclasses):
               ToAddImg = imgArrays[numclass][image]

               if len(np.shape(ToAddImg))!=3:
                   ToAddImg = imgArrays[numclass][image]*pow(2,16)/1000
                   ToAddImg =   np.expand_dims(ToAddImg, axis = 2)
               nextimg+=1
               DataTensor[nextimg,:,:,:] = ToAddImg.astype(np.int16)
               
          
        return DataTensor
    
    def ToNPZ1D(self,tracearray,numclasses,datatype):
        totalNumElements = np.sum([len(x) for x in tracearray])
        if totalNumElements!=0:
            if datatype=='data':
                DataTensor = np.empty([totalNumElements,len(tracearray[0][0]),1],dtype=np.int16)
            if datatype=='label':
                DataTensor = np.empty([totalNumElements,len(tracearray[0][0])],dtype=np.int16)
    
    
    
            AllTraces = []
            for Genome in tracearray:
                AllTraces.extend(Genome)
                
            random.Random(452856).shuffle(AllTraces)
            for i in range(0,len(AllTraces)):
                if datatype=='data':
                    DataTensor[i,:,:] = np.reshape(AllTraces[i],[len(AllTraces[i]),1])
                    
                if datatype=='label':
                    DataTensor[i,:] = AllTraces[i]
        else:
            DataTensor = np.empty([0,])

            
        return DataTensor
  
        

    
    
    
    

    
    
    
        
        
        
        
        
    
                
            
            
            
            
        
        
        
        
        
        
    
    


# Dt = DataLoader()          
# Dt.PrepareTrainingData('D:\Vibrio Harveyi\FOVData\CroppedAndInverted')
# Dt.PrepareLabeledData('D:\Vibrio Harveyi\FOVData\Mask')
  