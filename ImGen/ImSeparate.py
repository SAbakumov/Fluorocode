# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 18:35:31 2021

@author: Boris
"""
import os
import numpy as np
import PIL.Image as Image
from aicspylibczi import CziFile

############# INPUT ###############
path = r"D:\Elizabete\2021\VibrioHarveiiWidefieldCirculomics-Elizabete\Puragen\18-3ul"
name = "Image 18_3uL sample.czi"
threshold = 1000
############# INPUT ###############





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
  



