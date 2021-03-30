# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 18:35:31 2021

@author: Boris
"""
import os
import numpy as np
import PIL.Image as Image
from aicspylibczi import CziFile
import javabridge
import bioformats
import pathlib
############# INPUT ###############
path = r"D:\Elizabete\2021\VibrioHarveiTienen-2021-18-02_WaterNA0.2-MagBeads"
name = "VibrioHarveiWater-StretchedPerseus.czi"
threshold =2000
xsize = 1024 #Only for Olympus .ets files
############# INPUT ###############





def main(path, name, xsize, threshold):
    ext = pathlib.Path(name).suffix
    if ext == ".czi":
        GetFromCZI(path, name, threshold)
    if ext == ".ets":
        GetFromETS(path, name, xsize,xsize, threshold)

def PrintChannelInformation(dimensions, valid_dim):
    print("Processing SIM/WF images with properties:")
    if "C" in valid_dim:
        print("Channels: " + str(dimensions["C"][1]))
    if "H" in valid_dim:
        print("Phases: " + str(dimensions["H"][1]))
    if "R" in valid_dim:
        print("Rotations: " + str(dimensions["R"][1]))
    if "M" in valid_dim:      
        print("Scans: " + str(dimensions["M"][1]))
    if "X" in valid_dim:
        print("X-size: " + str(dimensions["X"][1]))
    if "Y" in valid_dim:
        print("Y-size: " + str(dimensions["Y"][1]))


def GetFromETS(path, name, xsize,ysize, threshold):

    javabridge.start_vm(class_path=bioformats.JARS)
    img = bioformats.load_image(os.path.join(path,name))*65536
    savepath = os.path.join(path, "Export")
    javabridge.kill_vm()

    XLimit = np.floor_divide( img.shape[0],xsize)*xsize
    YLimit = np.floor_divide( img.shape[1],ysize)*ysize
    try:
        os.makedirs(os.path.join(savepath,str(0)))
    except:
        print("Output directory exists for channel "+ str(0)+ ", overwriting...")
    nex = 0
    for x in range(0,XLimit,xsize):
        for y in range(0,YLimit,xsize):
            widefieldimg = img[x:x+xsize,y:y+ysize]
            widefieldimg[np.where(widefieldimg>threshold)]=threshold
            widefieldimg = widefieldimg/threshold*255   

            im = Image.fromarray(widefieldimg.astype(np.uint8))
            im.save(os.path.join(savepath,str(0) ,"Export_"+str(0)+"_"+ str(nex)+ '.tif'))
            nex= nex+1            




def GetFromCZI(path, name, threshold):
    img = CziFile(os.path.join(path, name))
    savepath = os.path.join(path, "Export")


    dimensions = img.dims_shape()[0]
    PrintChannelInformation(dimensions, img.dims)





    for channel in range(0,dimensions["C"][1]):
        nex = 0
        nrotations = 1
        nphases    = 1
        try:
            os.makedirs(os.path.join(savepath,str(channel)))
        except:
            print("Output directory exists for channel "+ str(channel)+ ", overwriting...")
            
        if "H" in img.dims:
            nphases   =dimensions["H"][1]
        if "R" in img.dims:
            nrotations=dimensions["R"][1]
            
        for image in range(0, dimensions["M"][1]):
            widefieldimg = np.zeros( [dimensions["X"][1], dimensions["Y"][1]])
            
            for phase in range(0,nphases):
                for rotation in range(0,nrotations):
                    img_cropped,shp = img.read_image(C = channel, H=phase,R=rotation,M=image)
                    widefieldimg = widefieldimg + np.squeeze(img_cropped)
            widefieldimg = widefieldimg/(nphases*nrotations)
            widefieldimg[np.where(widefieldimg>threshold)]=threshold
            widefieldimg = widefieldimg/threshold*255
            
            im = Image.fromarray(widefieldimg.astype(np.uint8))

            im.save(os.path.join(savepath,str(channel) ,"Export_"+str(channel)+"_"+ str(nex)+ '.tif'))
            nex= nex+1
    

main(path, name, xsize, threshold)


