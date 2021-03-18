#%%
import numpy as np
import tifffile as tiff
import os
import matplotlib.pyplot as plt
import skimage
from skimage.measure import regionprops
from Fit2DGauss import getFWHM_GaussianFitScaledAmp
import copy

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


MaskFolder = r'D:\Elizabete\2021\RAW_SIM_Lambda_MTC35_MbapI_561nm\Mask'
ImFolder = r'D:\Elizabete\2021\RAW_SIM_Lambda_MTC35_MbapI_561nm\Data'
Images = os.listdir(MaskFolder)
#%%
# for image in Images:
ths = 40
amps = []
bgs = []

for image in Images:
    if image.endswith('.tif'):
        FileName = image.replace( 'mask-','')
        
        print(str(FileName) + ' done')
        
        imMask =( rgb2gray( tiff.imread(os.path.join(MaskFolder,image)))*10).astype(np.int)
        im =  tiff.imread(os.path.join(ImFolder , FileName))


        imMask[imMask!=2549]=0
        imMask[imMask!=0]=1
        im = im*imMask
        imOr = copy.deepcopy(im)
        im[im<ths] =0 
        im[im>=ths] =1
        all_labels = regionprops(skimage.measure.label(im))

        for label in all_labels:
            if label.area>10 and label.area<20:
                y0, x0 = label.centroid
                x0  =np.int64(x0)
                y0  =np.int64(y0)
                if x0>20 and y0>20:
                    msk = imOr[y0-10:y0+10,x0-10:x0+10]
                    
                    if len(np.where(msk==0)[0])==0:
                        msk = imOr.flatten()
                        bg = np.mean(np.delete(msk,np.where(msk==0)[0]))
                        Gauss = imOr[y0-10:y0+10,x0-10:x0+10]
                        try:
                            FitRes = getFWHM_GaussianFitScaledAmp(Gauss- bg)
                            bgs.append(bg)
                            amps.append(FitRes[2])
                        except Exception as e:
                            print(e)


# %%
plt.rcParams.update({'font.size': 22})
plt.close('all')
plt.figure(figsize = (12,8))
hist, bins = np.histogram(amps,30)
plt.fill_between(bins[1:],hist,color = "blue",alpha=0.25)
plt.xlabel("Dye amplitude")
plt.ylabel("Counts")
plt.ylim([0, np.max(hist)+200])
print(np.mean(amps))
# %%
