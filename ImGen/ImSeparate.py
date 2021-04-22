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
import czifile as cz
import tifffile 
import re
import json
############# INPUT ###############
path = r"F:\Elizabete\20210401\_FOV____07_\stack1"
name = "frame_t_0.ets"
threshold =65000    # MAX YOU CAN PUT IS 65536
renormalized =False # Disabling renormalization will remove any thresholding and store original images split up as is



#####Only for Olympus .ets files###
xsize = 512 
NA = 1.45 
EM = 590
PX = 107
SIM = False 
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
            if renormalized:
                widefieldimg[np.where(widefieldimg>threshold)]=threshold
                widefieldimg = widefieldimg/threshold*255
                im = widefieldimg.astype(np.uint8)
            else:
                widefieldimg = widefieldimg-10
                im = widefieldimg.astype(np.uint16)
            
            # im = Image.fromarray(widefieldimg.astype(np.uint8))
            metadata = dict(NA=NA, EM=EM, PX=PX*10**(-9),SIM = SIM,threshold = threshold)

            tifffile.imsave( os.path.join(savepath,str(0) ,"Export_"+str(0)+"_"+ str(nex)+ '.tif'), im,imagej=True, metadata=metadata)
            nex= nex+1            




def GetFromCZI(path, name, threshold):
    img = CziFile(os.path.join(path, name))
    with cz.CziFile(os.path.join(path, name)) as czi:
        xml_metadata = czi.metadata()
    em_wavelength = re.findall(r'<EmissionWavelength>(.+?)</EmissionWavelength>',xml_metadata)
    lens_NA = re.findall(r'<LensNA>(.+?)</LensNA>',xml_metadata)
    pixelscaling = re.findall(r'<Distance Id="X">\n          <Value>(.+?)</Value>',xml_metadata)



    prefix = name[:-4]
    with open('dump.txt','w') as rb:
        rb.write(xml_metadata)
        rb.close()
    savepath = os.path.join(path, prefix+"Export")


    dimensions = img.dims_shape()[0]
    PrintChannelInformation(dimensions, img.dims)





    for channel in range(0,dimensions["C"][1]):
        if len(lens_NA)<=channel:
            lens_NA.append(lens_NA[0])
        if len(em_wavelength)<=channel:
            em_wavelength.append(em_wavelength[0])
        if len(pixelscaling)<=channel:
            pixelscaling.append(pixelscaling[0])
        if float(pixelscaling[channel])*10**9<40:
            SIM = True
        else:
            SIM = False

        nex = 0
        nrotations = 1
        nphases    = 1
        try:
            os.makedirs(os.path.join(savepath,str(channel),'Mask'))
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
            if renormalized:
                widefieldimg[np.where(widefieldimg>threshold)]=threshold
                widefieldimg = widefieldimg/threshold*255
                im = widefieldimg.astype(np.uint8)
            else:
                im = widefieldimg.astype(np.uint16)
               



            metadata = dict(NA=lens_NA[channel], EM=em_wavelength[channel], PX=pixelscaling[channel],SIM = SIM,threshold = threshold)
            tifffile.imsave( os.path.join(savepath,str(channel) ,"Export_"+str(channel)+"_"+ str(nex)+ '.tif'), im,imagej=True, metadata=metadata)



 
            nex= nex+1
    

main(path, name, xsize, threshold)


