# -*- coding: utf-8 -*-

import tifffile as tiff
import numpy as np
import cv2
import math
import scipy
import copy
from skimage.measure import regionprops
from skimage.measure import profile_line
import skimage
import csv
import os
import matplotlib.pyplot as plt 

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def get_rotation_angle(im):
    im=im.astype(np.uint8)
    im[im==245]=0
    im[im!=0]=1
    lines = cv2.HoughLinesP(im, 1, math.pi / 180.0, 100, minLineLength=10, maxLineGap=5)
    angles = []
    
    for [[x1, y1, x2, y2]] in lines:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)
        
    angle = np.mean(angles)
    return angle




def extract_rotated_traces(MaskFolder, ImFolder,csv_path):
    FullTraceArray = []
    Images = os.listdir(MaskFolder)
    
    for image in Images:
        try:
            if image.endswith('.tif'):
                FileName = image.replace( 'mask-','')
                
                print(str(FileName) + ' done')
                
                im =( rgb2gray( tiff.imread(os.path.join(MaskFolder,image)))*10).astype(np.int)
    
                imOr =  tiff.imread(os.path.join(ImFolder , FileName))
                FileName = image.replace( '.tif','.png')
    
                
                Traces = []    
                angle  = get_rotation_angle(im)    
                reg = regionprops(im.astype(np.int))
                # if angle>0:
                #     angle = -angle
                
                
                
    
                for det in range(0,len(reg)-1):
                        detected = reg[det]
                        color = detected.label
                        bbox = detected.bbox
                
                        label_img = copy.deepcopy(im)
                        label_img[np.where(label_img!=color)]=0
                        label_img[np.where(label_img!=0)]=1
                        label_img = label_img.astype(np.int8)
                        
                        detectedPart =label_img[bbox[0]:bbox[2],bbox[1]:bbox[3]]
                        detectedPartOr =imOr[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        
                        detectedPartPadded = np.pad(detectedPart,((0, 0), (detectedPart.shape[0], detectedPart.shape[0])) , 'constant')    
                        detectedRot = scipy.ndimage.rotate(detectedPartPadded,angle,reshape=False)
          
                        for i in range(0 , detectedRot.shape[0]):
                            detectedRot[:,i*15:i*15+10] = 0
                        all_labels = regionprops(skimage.measure.label(detectedRot))
                        pivotpoints = []
                        for det_label in all_labels:
                            y0 ,x0  = det_label.centroid;
                            pivotpoints.append(np.array([x0,y0]))
                        
                        
                        pivotpoints  = np.array(pivotpoints)
                        pivotpoints[:,0] = pivotpoints[:,0]- detectedRot.shape[1]/2
                        pivotpoints[:,1] = pivotpoints[:,1]- detectedRot.shape[0]/2
                      
                        
                        a  = pivotpoints
                        a = a[a[:,0].argsort()]
                        c, s = np.cos(- np.radians(angle)), np.sin(- np.radians(angle))
                        R = np.array(((c, -s), (s, c)))
                        a = a.dot(R)
                        
                        a[:,0] = a[:,0] + detectedPart.shape[1]/2
                        a[:,1] = a[:,1] + detectedPart.shape[0]/2
                        allprof = np.zeros(0)
                        
                        
                        for i in range(0,len(a)-1):
                            p = profile_line(detectedPartOr, (a[i,1], a[i,0]), (a[i+1,1], a[i+1,0]))
                            allprof = np.concatenate([allprof,p[1:]])
                            
                        plt.imshow(detectedPartOr)
                        Traces.append(allprof)
                        plt.imshow(imOr)
                        plt.plot(a[:,0]+bbox[1],a[:,1]+bbox[0],color='red')
                if any([len(x) for x in Traces])==0:
                    FullTraceArray = FullTraceArray+Traces
                FullTraceArray = FullTraceArray+Traces
                plt.savefig(os.path.join(r"D:\Elizabete\2021\VibrioHarveiiWidefieldCirculomics-Elizabete\Puragen\18-3ul\Processed",FileName))
    
                plt.close('all')
                
            with open(csv_path, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                for k in FullTraceArray:
                    writer.writerow( k[:])
        except:
            print("Failed")
                

if __name__ == '__main__':
    
    ########### INPUT ############
    MaskFolder = r'D:\Elizabete\2021\VibrioHarveiiWidefieldCirculomics-Elizabete\Puragen\18-3ul\Processed\Extracted Traces'
    ImFolder   = r'D:\Elizabete\2021\VibrioHarveiiWidefieldCirculomics-Elizabete\Puragen\18-3ul\Processed'
    csv_path = r'D:\Elizabete\2021\VibrioHarveiiWidefieldCirculomics-Elizabete\Puragen\18-3ul\Processed\segmented_traces_averages.csv'
    
    extract_rotated_traces(MaskFolder, ImFolder,csv_path)
    ##############################
