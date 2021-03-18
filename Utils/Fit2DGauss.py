import numpy as np
import scipy.optimize as opt

def twoDGaussianScaledAmp( val, xo, yo, sigma_x, sigma_y, amplitude, offset):
    x = val[0]
    y = val[1]
    xo = float(xo)
    yo = float(yo)    
    g = offset + amplitude*np.exp( - (((x-xo)**2)/(2*sigma_x**2) + ((y-yo)**2)/(2*sigma_y**2)))
    return g.ravel()


def getFWHM_GaussianFitScaledAmp(img):

    x = np.linspace(0, img.shape[1], img.shape[1])
    y = np.linspace(0, img.shape[0], img.shape[0])
    x, y = np.meshgrid(x, y)
    #Parameters: xpos, ypos, sigmaX, sigmaY, amp, baseline
    initial_guess = (img.shape[1]/2,img.shape[0]/2,3,3,1,2)
    # subtract background and rescale image into [0,1], with floor clipping
    bg = np.percentile(img,5)
    # img_scaled = np.clip((img - bg) / (img.max() - bg),0,1)
    img_scaled =img
    popt, pcov = opt.curve_fit(twoDGaussianScaledAmp, [x, y], 
                               img_scaled.ravel(), p0=initial_guess,
                               bounds = ((img.shape[1]*0.4, img.shape[0]*0.4, 1, 1, 0.5, -2),
                                     (img.shape[1]*0.6, img.shape[0]*0.6, img.shape[1]/2, img.shape[0]/2, 100,40)))
    xcenter, ycenter, sigmaX, sigmaY, amp, offset = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
    FWHM_x = np.abs(4*sigmaX*np.sqrt(-0.5*np.log(0.5)))
    FWHM_y = np.abs(4*sigmaY*np.sqrt(-0.5*np.log(0.5)))
    return (FWHM_x, FWHM_y,amp,offset)

