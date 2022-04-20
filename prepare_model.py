import numpy as np
from numpy import ma

import sys
import os
import time

from scipy import signal
from scipy import interpolate
from scipy.optimize import curve_fit
import scipy.stats as stats


#import astropy
#from astropy.io import fits
#from astropy.io import fits
#from astropy.stats import sigma_clip
#from astropy.modeling import models, fitting, polynomial


import batman
import matplotlib.pyplot as plt

import convolve as conv


def strided_app(a, L, S ):  
    # This function is an updated moving median 
    # used in the normalization process
    # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

class reduced:

    def __init__(self,nord,W,Rp,R_s):

        self.ord  = nord  ## List of orders to be used -- after preselection

        self.Wm = W
        self.Rp = Rp
        self.R_s = R_s

    def convolve(self,rot_speed,superrot):

         R = self.Rp
         #We need frequency in the convolution, it's easier
         freq = 299792458.0/(self.Wm*1e-9)

         rot_freq,rot_Rp = conv.rotate(R,freq,rot_speed,superrot)
         self.Wm = rot_freq
         self.Rp = rot_Rp
         
    def normalize(self,nf=501):     
        #renor
        out = np.zeros((2,np.shape(self.Wm)[0]))
        out[0] = self.Wm
        out[1] = (self.Rp/self.R_s)**2


        win = np.percentile(strided_app(out[1],nf,1),0.5, axis=-1)
        try :
            out_final = np.zeros((2,len(out[1])-nf+1))
        except:
            print ("Removed order : ",self.ord)
            exit
        out_final[0]  = out[0][int((nf-1)/2):-int((nf-1)/2)]
        out_final[1] = win - out[1][int((nf-1)/2):-int((nf-1)/2)]

        self.Wm = out_final[0]
        self.Rp = out_final[1]


        



def prepare(model_dic,R_s,orderstot,winds=False,rot_speed=0.0,superrot=0.0):

    models = []
    wavelength = model_dic["freq_nm"]
    radius = model_dic["radius_transm"]
    for i in range(len(orderstot)):
        ### First we have to select only the portion of  the model that is of interest for us
        # limits = [self.lambdas[no][0]*0.995,self.lambdas[no][1]*1.005]
        no = orderstot[i]
        M  = reduced(no,wavelength[i],radius[i],R_s)
        

        if winds:
            M.convolve(rot_speed,superrot)
        
        # plt.plot(wavelength[i],radius[i])
        
        M.normalize()

        models.append(M)

    

    return {
        "models": models
        }











