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
import json
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
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    for image in Images:
        try:
            if image.endswith('.tif'):
                FileName = image.replace( 'mask-','')
                print(str(FileName) + ' done')
                with tiff.TiffFile( os.path.join(ImFolder , FileName)) as tif:
                    metadata = tif.imagej_metadata


                im =( rgb2gray( tiff.imread(os.path.join(MaskFolder,image)))*10).astype(np.int64)+10
    
                imOr =  tiff.imread(os.path.join(ImFolder , FileName))
                FileName = image.replace( '.tif','.png')
    
                # print(np.unique(im.astype(np.int64)))
                Traces = []    
                # angle  = get_rotation_angle(im)    
                reg = regionprops(im.astype(np.int64))
                # if angle>0:
                #     angle = -angle
                
                
                
    
                for det in range(0,len(reg)):
                    if reg[det].label!=2559:
                        detected = reg[det]
                        color = detected.label
                        bbox = detected.bbox
                        angle = 90-math.degrees(detected.orientation)


                        
                        label_img = copy.deepcopy(im)
                        label_img[np.where(label_img!=color)]=0
                        label_img[np.where(label_img!=0)]=1
                        label_img = label_img.astype(np.int8)
                        
                        detectedPart =label_img[bbox[0]:bbox[2],bbox[1]:bbox[3]]
                        detectedPartOr =imOr[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        
                        detectedPartPadded = np.pad(detectedPart,((0, 0), (detectedPart.shape[0], detectedPart.shape[0])) , 'constant')    
                        detectedRot = scipy.ndimage.rotate(detectedPartPadded,angle,reshape=False)
                        all_labels = regionprops(skimage.measure.label(detectedRot))
                        if len(all_labels)==1 :
                            
                            for i in range(0 , detectedRot.shape[0]):
                                detectedRot[:,i*20:i*20+15] = 0
                            all_labels = regionprops(skimage.measure.label(detectedRot))
                            pivotpoints = []
                            for det_label in all_labels:
                                y0 ,x0  = det_label.centroid
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
                            plt.figure(figsize=(18,8))
                            plt.subplot(1,2,1)
                            for i in range(0,len(a)-1):
                                p = profile_line(detectedPartOr, (a[i,1], a[i,0]), (a[i+1,1], a[i+1,0]),order=5,mode='constant')
                                allprof = np.concatenate([allprof,p[1:]])
                            if len(allprof)>0:
                                plt.imshow(detectedPartOr)
                                plt.subplot(1,2,2)
                                plt.plot(allprof)
                                plt.savefig('foo.png')
                                plt.close('all')      
                                Traces.append(allprof)
                            # plt.imshow(imOr)
                            # plt.plot(a[:,0]+bbox[1],a[:,1]+bbox[0],color='red')
                if any([len(x) for x in Traces])==0:
                    FullTraceArray = FullTraceArray+Traces
                FullTraceArray = FullTraceArray+Traces
    
                plt.close('all')
                    


                with open(os.path.join(csv_path, 'segmented_traces_averages.csv'),mode='w', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                    for k in FullTraceArray:
                        writer.writerow( k[:])
                with open( os.path.join(csv_path,'config.json'), 'w') as fp:
                    json.dump(metadata, fp)
        except Exception as e:
            print("Failed")
            print(e)

                

if __name__ == '__main__':
    
    ########### INPUT ############
    # folders = ['SIM-1Export' ,'SIM-2Export','SIM-4Export','SIM-6Export','SIM-8Export','SIM-1-NoBaselineCutExport','SIM-2-NoBaselineCutExport','SIM-4-NoBaselineCutExport','SIM-6-NoBaselineCutExport','SIM-8-NoBaselineCutExport']
    folders = ['Export']
    base = r'D:\Elizabete\2021\20210420_CellSens\_FOV_017_\stack1'
    for folder in folders:
        MaskFolder = os.path.join(base, folder,'0\Mask')
        ImFolder   =  os.path.join(base, folder,'0')
        csv_path = os.path.join(base,folder,'0\Results')
        
        extract_rotated_traces(MaskFolder, ImFolder,csv_path)

    ##############################
