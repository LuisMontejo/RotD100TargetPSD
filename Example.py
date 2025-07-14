'''
luis.montejo@upr.edu

Example from article.

Montejo, L.A. 2025. "Generation of Fourier Amplitude Spectra and Power Spectral 
Density Functions Compatible with Orientation-Independent Design Spectra" 
Earthquake Spectra (submiited for publication)
'''

import numpy as np
from TargetRotDnnPSD_Module import RotDnnTargetFASPSD

if __name__ == '__main__':
    
    g = 9.81 # m.s2
    
    # load target response spectrum:
         
    dspec = np.loadtxt('BSSA14_M7_VS400_RJB50_RotD100.txt') # file with the target response speectrum
    f_or = dspec[:,0]                   # frequencies [Hz]
    ds_or = dspec[:,1]                  # amplitudes [g]
    sd575 = 10.4                        # expected strong motion duration (SD5-75)
    freqs_des = np.geomspace(0.1,98,50) # frequencies where the output target 
                                        # FAS and PSD would be interpolated
    
    TargetFAS,TargetPSD=RotDnnTargetFASPSD(f_or,ds_or,sd575,freqs_des,workname='RotDnnTargetFASPSD_targetmean', 
                                           nnPSA=100,nnPSD=100,sdratio=1.3, 
                                           smo=2,b=20,
                                           zi=0.05,F1=0.1,F2=100,
                                           allow_err=2.5,neqsPSD=1000,plots=1)