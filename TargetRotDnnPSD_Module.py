'''
TargetRotDnnPSD_Module
luis.montejo@upr.edu
Generation of Fourier Amplitude Spectra and Power Spectral Density Functions 
Compatible with Orientation-Independent Design Spectra

===============================================================================
References:
    
Montejo, L.A. 2025. "Generation of Fourier Amplitude Spectra and Power Spectral 
Density Functions Compatible with Orientation-Independent Design Spectra" 
Earthquake Spectra (submiited for publication)
    
Montejo, L.A. 2024. "Strong-Motion-Duration-Dependent Power Spectral Density 
Functions Compatible with Design Response Spectra" Geotechnics 4, no. 4: 1048-1064. 
https://doi.org/10.3390/geotechnics4040053

Montejo L.A.; Vidot-Vega, A.L.  2017. “An Empirical Relationship between Fourier 
and Response Spectra Using Spectrum-Compatible Times Series” Earthquake Spectra; 
33 (1): 179–199. doi: https://doi.org/10.1193/060316eqs089m

Chi-Miranda M.; Montejo, L.A. 2018. “FAS-Compatible Synthetic Signals for 
Equivalent-Linear Site Response Analyses” Earthquake Spectra; 34 (1): 377–396. 
doi: https://doi.org/10.1193/102116EQS177M

===============================================================================

This module contains the Python functions required to generate strong motion 
duration dependent target RotDnn FAS and PSD functione compatible with a RotDnn
design/target response spectrum. The functions included can also be used to 
compute rotated ROtDnn FAS, PSD and PSA.
'''

def load_PEERNGA_record(filepath):
    '''
    Load record in .at2 format (PEER NGA Databases)

    Input:
        filepath : file path for the file to be load
        
    Returns:
    
        acc : vector wit the acceleration time series
        dt : time step
        npts : number of points in record
        eqname : string with year_name_station_component info

    '''

    import numpy as np

    with open(filepath) as fp:
        line = next(fp)
        line = next(fp).split(',')
        year = (line[1].split('/'))[2]
        eqname = (year + '_' + line[0].strip() + '_' + 
                  line[2].strip() + '_comp_' + line[3].strip())
        line = next(fp)
        line = next(fp).split(',')
        npts = int(line[0].split('=')[1])
        dt = float(line[1].split('=')[1].split()[0])
        acc = np.array([p for l in fp for p in l.split()]).astype(float)
    
    return acc,dt,npts,eqname

def logfrequencies(start_freq, end_freq, points_per_decade):
    
    '''
    Calculates logarithmically spaced frequencies within a given range.

    Parameters
    ----------
        start_freq (float): The starting frequency in Hz.
        end_freq (float): The ending frequency in Hz.
        points_per_decade (int): The desired number of points per frequency decade.

    Returns
    -------
        numpy.ndarray: An array of logarithmically spaced frequencies.
    '''
    import numpy as np
    
    num_decades = np.log10(end_freq / start_freq)  # Number of decades in the range
    total_points = int(num_decades * points_per_decade + 1)  # Ensure at least 100 points/decade

    # Use logspace to create logarithmically spaced frequencies
    frequencies = np.logspace(np.log10(start_freq), np.log10(end_freq), total_points)

    return frequencies

def saragoni_hart_w(npoints,eps=0.25,n=0.4,tn=0.6):
    '''
    returns a Saragoni-Hart type of window 

    Parameters
    ----------
    npoints : integer
        number of points to generate the window
    eps : float (0-1), optional
        relative distance/time at which the amplitude reaches 1. The default is 0.25.
    n : float (0-1), optional
        relative amplitude at tn. The default is 0.4.
    tn : float (eps,1], optional
        relative distance/time at which the amplitude reaches n. The default is 0.6.

    Returns
    -------
    w : Saragoni-Hart window (1D array)

    '''
    import numpy as np
    
    b = -(eps*np.log(n))/(1+eps*(np.log(eps)-1))
    c = b/eps
    a = (np.exp(1)/eps)**b
    t = np.linspace(0,1,npoints)
    w = a*(t/tn)**b*np.exp(-c*(t/tn))
    
    return w

def FASPSAratio(f,sd575):
    '''
    Target FAS based on empirical relationship between Fourier and 
    response spectra (Montejo & Vidot-Vega, 2017)

    Parameters
    ----------
    f : 1D array
        freqeuncies [Hz] where the ratios are compted
    sd575 : float
        target sd5-75 [s]

    Returns
    -------
    ratio : 1D array
        ratios between PSA and FAS at each frequency in f

    '''

    aa75 = 0.0512
    ab75 = 0.4920
    ac75 = 0.1123
    ba75 = -0.5869
    bb75 = -0.2650
    bc75 = -0.4580
    ratio = (aa75*sd575**ab75+ac75)*f**(ba75*sd575**bb75+bc75)
    
    return ratio

def log_interp(x, xp, fp):
    '''
    Performs logarithmic interpolation

    Parameters
    ----------
    x : 1D array
        The x-coordinates at which to evaluate the interpolated values.
    xp : 1D array
        The x-coordinates of the data points.
    fp : 1D array
        The y-coordinates of the data points

    Returns
    -------
    f  : 1D array
        The interpolated data
    '''
    import numpy as np
    with np.errstate(divide="ignore"):
        logx = np.log10(x)
        logxp = np.log10(xp)
        logfp = np.log10(fp)
        f = np.power(10.0, np.interp(logx, logxp, logfp))
    return f

def RSFDtheta(T,s1,s2,z,dt,theta):
    '''
    Rotated response spectra in the frequency domain, 
    returns the spectra for each angle accommodated in 2D arrays

    Parameters
    ----------
    T : 1D array
        periods defining the spectra
    s1 : 1D array
        acceleration series in H0
    s2 : 1D array
        acceleration series in H90 
    z : float
        damping ratio for spectra
    dt: float
        time step [s]
    theta : 1D array
        rotation angles

    Returns
    -------
    PSA : 2d array
        rotated PSA
    PSV : 2d array
        rotated PSV
    SD : 2d array
        rotated SD

    '''
    import numpy as np
    from numpy.fft import rfft, irfft,rfftfreq
    
    pi = np.pi
    theta = theta*pi/180
    
    ntheta = np.size(theta)
    npo = np.max([np.size(s1),np.size(s2)])
    nT  = np.size(T)
    
    SD  = np.zeros((ntheta,nT))
    
    n = int(2**np.ceil(np.log2(npo+4*np.max(T)/dt)))  # add zeros to provide enough quiet time
    
    s1 = np.append(s1,np.zeros(n-npo))
    s2 = np.append(s2,np.zeros(n-npo))
       
    freqs = rfftfreq(n,d=dt)
    ww    = 2*pi*freqs                      # vector with frequencies [rad/s]
    ffts1 = rfft(s1) 
    ffts2 = rfft(s2) 
    
    m = 1
    for kk in range(nT):
        w = 2*pi/T[kk] ; k=m*w**2; c = 2*z*m*w
        
        H1 = 1       / ( -m*ww**2 + k + 1j*c*ww )  # Transfer function (half) - Receptance
               
        CoFd1 = H1*ffts1   # frequency domain convolution
        d1 = irfft(CoFd1)   # go back to the time domain (displacement)
        
        CoFd2 = H1*ffts2   # frequency domain convolution
        d2 = irfft(CoFd2)   # go back to the time domain (displacement)
        
        Md1,Mtheta = np.meshgrid(d1,theta,sparse=True, copy=False)
        Md2,_      = np.meshgrid(d2,theta,sparse=True, copy=False)
                
        drot = Md1*np.cos(Mtheta)+Md2*np.sin(Mtheta)
        
        SD[:,kk] = np.max(np.abs(drot),axis=1)
            
    PSV = (2*pi/T)* SD
    PSA = (2*pi/T)**2 * SD
    
    return PSA,PSV,SD

def RSPWtheta(T,s1,s2,z,dt,theta):
    '''
    Rotated response spectra using piecewise, 
    returns the spectra for each theta accommodated in 2D arrays

    Parameters
    ----------
    T : 1D array
        periods defining the spectra
    s1 : 1D array
        acceleration series in H0
    s2 : 1D array
        acceleration series in H90 
    z : float
        damping ratio for spectra
    dt: float
        time step [s]
    theta : 1D array
        rotation angles

    Returns
    -------
    PSA : 2d array
        rotated PSA
    PSV : 2d array
        rotated PSV
    SD : 2d array
        rotated SD
    '''
    import numpy as np  
    
    pi = np.pi
    theta = theta*pi/180
    ntheta = np.size(theta)
    
    nT    = np.size(T)						      # number of natural periods
    SD  = np.zeros((ntheta,nT))
    n1 = np.size(s1); n2 = np.size(s2)
    
    if n1>n2:
        n = n2
        s1 = s1[:n]
    else:
        n = n1
        s2 = s2[:n]
    
    for k in range(nT):
       wn = 2*pi/T[k]
       wd = wn*(1-z**2)**(1/2)
       
       u1 = np.zeros((2,n))          # matrix with velocities and displacements
       u2 = np.zeros((2,n))          # matrix with velocities and displacements
       
       ex = np.exp(-z*wn*dt)
       cwd = np.cos(wd*dt)
       swd = np.sin(wd*dt)
       zisq = 1/(np.sqrt(1-(z**2)))
    
       a11 = ex*(cwd+z*zisq*swd)
       a12 = (ex/wd)*swd
       a21 = -wn*zisq*ex*swd
       a22 = ex*(cwd-z*zisq*swd)
    
       b11 = ex*(((2*z**2-1)/((wn**2)*dt)+z/wn)*(1/wd)*np.sin(wd*dt)+
           (2*z/((wn**3)*dt)+1/(wn**2))*np.cos(wd*dt))-2*z/((wn**3)*dt)
       b12 = -ex*(((2*z**2-1)/((wn**2)*dt))*(1/wd)*np.sin(wd*dt)+
           (2*z/((wn**3)*dt))*np.cos(wd*dt))-(1/(wn**2))+2*z/((wn**3)*dt)
       b21 = -((a11-1)/((wn**2)*dt))-a12
       b22 = -b21-a12
       
       A = np.array([[a11,a12],[a21,a22]])
       B = np.array([[b11,b12],[b21,b22]])
    
       for q in range(n-1):
          u1[:,q+1] = np.dot(A,u1[:,q]) + np.dot(B,np.array([s1[q],s1[q+1]]))
          u2[:,q+1] = np.dot(A,u2[:,q]) + np.dot(B,np.array([s2[q],s2[q+1]]))
       
       d1 = u1[0,:]; d2 = u2[0,:]
       
       Md1,Mtheta = np.meshgrid(d1,theta,sparse=True, copy=False)
       Md2,_      = np.meshgrid(d2,theta,sparse=True, copy=False)
                
       drot = Md1*np.cos(Mtheta)+Md2*np.sin(Mtheta) 
       SD[:,k] = np.max(np.abs(drot),axis=1)   
       

    
    PSV = (2*pi/T)*SD                    # pseudo-vel. spectrum
    PSA = (2*pi/T)**2 *SD  	             # pseudo-accel. spectrum
    
    return PSA, PSV, SD

def ResponseSpectrumTheta(T,s1,s2,z,dt,theta):
    '''
    decides what approach to use to estimate 
    the response spectrum based on damping value 
    (>=4% frequency domain, <3% piecewise)

    Parameters
    ----------   
    T : 1D array
        periods defining the spectra
    s1 : 1D array
        acceleration series in H0
    s2 : 1D array
        acceleration series in H90 
    z : float
        damping ratio for spectra
    dt: float
        time step [s]
    theta : 1D array
        rotation angles
        
    Returns
    -------
    PSA : 1D array
        PSA
    PSV : 1D array
        PSV
    SD : 1D array
        SD
    '''
    if z>=0.03:
        PSA, PSV, SD = RSFDtheta(T,s1,s2,z,dt,theta)
    else:
        PSA, PSV, SD = RSPWtheta(T,s1,s2,z,dt,theta)
        
    return PSA, PSV, SD

def rotdnn(s1,s2,dt,zi,T,nn):
    '''
    Computes rotated and RotDnn spectra

    Parameters
    ----------
    s1 : 1D array
        acceleration series in H0
    s2 : 1D array
        acceleration series in H90 
    dt : float
        time step [s]
    zi : float
        damping ratio for spectra
    T : 1D array
        periods defining the spectra
    nn : int
        percentile for RotDnn

    Returns
    -------
    PSArotnn : 1D array
        PSARotDnn
    PSA180 : 2D array
        Rotated spectra [0-179 degrees]

    '''
    import numpy as np
    n1 = np.size(s1); n2 = np.size(s2); n = np.min((n1,n2))
    s1 = s1[:n]; s2 = s2[:n]
    theta = np.arange(0,180,1)
    PSA180,_,_,=ResponseSpectrumTheta(T,s1,s2,zi,dt,theta)
    PSArotnn = np.percentile(PSA180,nn,axis=0)
    return PSArotnn,PSA180

def FASPSAcomps2D(nt,dt,envelope1,envelope2,TargetFAS0,TargetFAS90,T,zi,nn):
    '''
    Generates a synthetic motion comprised of two horizontal 
    components given a target FAS and amplitude envelope for each 
    component. Returns the RoTDnn response spectrum of the motion.

    Parameters
    ----------
    nt : int
        number of points to generate the motion
    dt : float
        time between samples [s]
    envelope1 : 1D array
        amplitde envelope for H0
    envelope2 : 1D array
        amplitude envelope for H90
    TargetFAS0 : 1D array
        target FAS for H0
    TargetFAS90 : D array
        target FAS for H90
    T : 1D array
        periods for response spectrum 
    zi : float
        damping ratio for response spectrum
    nn : int
        percentile for RotDnn

    Returns
    -------
    PSArotnn : 1D array
         PSARotDnn

    '''
    import numpy as np
   
    so1 = np.random.randn(nt) 
    so2 = np.random.randn(nt) # generates the synthetic signals
    
    so1 = envelope1*so1
    so2 = envelope2*so2
    
    FSo1 = np.fft.rfft(so1)  # initial Fourier coefficients
    FSo2 = np.fft.rfft(so2)
    
    FASo1 = dt*np.abs(FSo1)  # initial FAS
    FASo2 = dt*np.abs(FSo2) 
    
    ff1 = TargetFAS0 / FASo1 # modification factors
    ff2 = TargetFAS90 / FASo2
    
    FS1 = ff1 * FSo1 # modified Fourier coefficients
    FS2 = ff2 * FSo2
    
    s1 = np.fft.irfft(FS1) # FAS compatible signal
    s2 = np.fft.irfft(FS2)
    
    PSArotnn,_=rotdnn(s1,s2,dt,zi,T,nn)
    
    return PSArotnn

def SignificantDuration(s,t,ival=5,fval=75):
    '''
    Estimates significant duration and Arias Intensity
    
    Parameters
    ----------
    s : 1d array
        acceleration time-history
    t : 1d array
        time vector
    ival : float, optional
        Initial % of Arias Intensity to estimate significant duration. 
        The default is 5.
    fval :float, optional
        Final % of Arias Intensity to estimate significant duration. 
        The default is 75.

    Returns
    -------
    sd : float
        significant duration
    AIcumnorm : 1d array
        normalized cummulative AI
    AI : float
        Arias Intensity (just the integral, 2*pi/g not included)
    t1 : float
        initial time for sd
    t2 : float
        final time for sd

    '''
    from scipy import integrate
    AIcum = integrate.cumulative_trapezoid(s**2, t, initial=0)
    AI = AIcum[-1]
    AIcumnorm = AIcum/AI
    t_strong = t[(AIcumnorm>=ival/100)&(AIcumnorm<=fval/100)]
    t1, t2 = t_strong[0], t_strong[-1]
    sd = t2-t1
    return sd,AIcumnorm,AI,t1,t2

def PSDFFTEq(so,fs,alphaw=0.1,duration=(5,75),nFFT='nextpow2',basefornFFT = 0,detrend='linear',smo=2,b=20,dsamp=0):
    
    '''
    Calculates the power spectral densityof earthquake acceleration time-series
    FFT is normalized by dy dt, FFT/PSD is calculated over the stong motion duration
    returns the one-sided PSD and a "smoothed" version using either Konno-Ohmachi
    or SRP3.7.1 guidelines.
    
    Parameters
    ----------
    so : 1D array
        acceleration time-series
        
    fs : integer
        sampling frequency
        
    alphaw : Optional, float, tukey window parameter [0 1], defaults to 0.1
             0 -> rectangular, 1 -> Hann
    
    duration: Optional, tuple or None
    
              (a,b) stong motion duration used to defined the portion of the signal 
              used to calculate FFT and PSD.Defined as the duration corresponding 
              to a a%-to-b% rise of the cumulative Arias energy
              
              None: the whole signal is used
              
              The default is (5,75).
        
    nFFT : Optional (defaults to 'nextpow2'), number of points to claculate the FFT, options :
        
        'nextpow2': zero padding until the mext power of 2 is reached
        
        'same': keep the number of points equal to the number of poitns in
                the signal
                
        An integer:  
            If n is smaller than the length of the input, the input is cropped. 
            If it is larger, the input is padded with zeros. 
        
    basefornFFT: Optional, interger 0 or 1, whether nFFT is determined based on
                 the original/total number of datapoints in the signal or based
                 on the strong motion part.
                 
                 0 -> total number, 1 -> strong motion part
                 
                 defaults to 0
                 
    detrend:(defaults to linear)
         None: no detrending is performed
        'linear' (default): the result of a linear least-squares fit to data is subtracted from data. 
        'constant': only the mean of data is subtracted
        
    smo: (integer, defaults to 2):
        0: no smoothing
        1: smooth using Konno-Ohmachi
        2: smooth using moving window of varibale width (NRC SRP3.7.1)
        
        
    b: (float) Used only if smo is 1 or 2:
       
       for smo1 (Konno-Ohmachi):
       coefficient for band width, common values are 188.5 (e.g. Kottke etal 2021)
       - provides a smoothing # operator with a bandwidth of 1/30 of a decade, and
       40 (e.g. Bora et al 2019) which results on a smoother spectrum
       
       for smo2: NRC SRP 3.7.1 reccomends 20 (20% of the central frequency)
    
    dsamp: defaults to 0 : no downsampling
           1D array with the frequencies to dowsample the spectrum

    Returns
    -------
    freqs :  Vector with the frequencies
    PSD  : One-sided power spectral density
    sd : duration used to calculated FFT/SD
    AI : Arias intensity of the signal (Just the integral, units depend on the
                                        initial signal units, pi/2g is not applied)
    
    optional:
    freqs_smooth: Vector with the frequencies for downsample
    PSD_smooth : One-sided average power spectral density
    '''
    
    import numpy as np
    from scipy import signal
    import pykooh
    
    no = np.size(so)
    dt = 1/fs
    t = np.linspace(0,(no-1)*dt,no)   # time vector 
      
    if duration==None:
        duration = (0,100)
        
    if len(duration)==2 :
        sd,AIcum,AI,t1,t2 = SignificantDuration(so,t,ival=duration[0],fval=duration[1])
        locs = np.where((t>=t1-dt/2)&(t<=t2+dt/2))
        nlocs = np.size(locs)
        s = so[locs]
        window = signal.windows.tukey(nlocs,alphaw)
        if detrend=='linear':
            s = signal.detrend(s,type='linear')
        elif detrend=='constant':
            s = signal.detrend(s,type='constant')
        elif detrend!=None:
            print('*** error defining detrend in PSDFFTEq function ***')
            return
        s = window*s
        
    else:
        print('*** error defining duration in PSDFFTEq function ***')
        return
    
    if basefornFFT == 0:
        n = no
    else:
        n = nlocs
        
    if nFFT=='nextpow2':
        nFFT = int(2**np.ceil(np.log2(n)))
    elif nFFT=='same':
        nFFT = n
    elif not isinstance(nFFT, int):
        print('*** error defining nFFT in PSDFFTEq function ***')
        return
        
    freqs = np.fft.rfftfreq(nFFT, d=dt)
    nfrs = len(freqs)
    Fs = np.fft.rfft(s,nFFT)
    mags = dt*np.abs(Fs)
    PSD = 2*mags**2/(2*np.pi*sd)
    
    
    if smo==0:
        return freqs,PSD,sd,AI         
       
    if smo==1:
        PSD_smooth_allfreq =  pykooh.smooth(freqs, freqs, PSD, b)
            
    elif smo==2:
        PSD_smooth_allfreq = np.copy(PSD)
        overl = b/100
        if overl>0:   
            for q in range(1,nfrs-1):
                lim1 = (1-overl)*freqs[q]
                lim2 = (1+overl)*freqs[q]
                
                if freqs[0]>lim1:
                    lim1 = freqs[0]
                    lim2 = freqs[q]+(freqs[q]-freqs[0])
                if freqs[-1]<lim2:
                    lim2 = freqs[-1]
                    lim1 = freqs[q]-(freqs[-1]-freqs[q])
                    
                locsf = np.where((freqs>=lim1)&(freqs<=lim2))
                PSD_smooth_allfreq[q]=np.mean(PSD[locsf])
    else:
        print('*** error defining smo in PSDFFTEq function ***')
        return
    
    if isinstance(dsamp,(float,int)):
        PSD_smooth = np.copy(PSD)
        dsampfreqs=np.copy(freqs)
    elif isinstance(dsamp,(list,tuple,np.ndarray)):
        dsampfreqs=np.array(dsamp)
        PSD_smooth = log_interp(dsampfreqs, freqs, PSD_smooth_allfreq)
    else:
        print('*** error defining dsamp in PSDFFTEq function ***')
        return

    return freqs,PSD,sd,AI,dsampfreqs,PSD_smooth

def PSDRotD(s1,s2,fs,nn,duration=(5,75),nFFT='nextpow2',smo=2,b=20,dsamp=0):
    '''
    Parameters
    ----------
    s1 : 1D array
        component 1 - acceleration time-series
    s2 : 1D array
        component 2 - acceleration time-series
    fs : integer
        sampling frequency
    
    nn : percentile, e.g. 50 for median, 100 for envelope
    
    duration : Optional, tuple or None
    
              (a,b) stong motion duration used to defined the portion of the signal 
              used to calculate FFT and PSD.Defined as the duration corresponding 
              to a a%-to-b% rise of the cumulative Arias energy
              
              None: the whole signal is used
              
              The default is (5,75).
              
    nFFT : Optional (defaults to 'nextpow2'), number of points to claculate the FFT, options :
        
        'nextpow2': zero padding until the mext power of 2 is reached
        
        'same': keep the number of points equal to the number of poitns in
                the signal
                
        An integer:  
            If n is smaller than the length of the input, the input is cropped. 
            If it is larger, the input is padded with zeros. 
    
    smo: (integer, defaults to 2):
        0: no smoothing
        1: smooth using Konno-Ohmachi
        2: smooth using moving window of varibale width (NRC SRP3.7.1)
        
        
    b: (float) Used only if smo is 1 or 2:
       
       for smo1:
       coefficient for band width, common values are 188.5 (e.g. Kottke etal 2021)
       - provides a smoothing # operator with a bandwidth of 1/30 of a decade, and
       40 (e.g. Bora et al 2019) which results on a smoother spectrum
       
       for smo2: NRC SRP 3.7.1 reccomends 20 (20% of the central frequency)
    
    dsamp: defaults to 0 : no downsampling
           1D array with the frequencies to dowsample the spectrum

    Returns
    -------
    returns:
        freqs,SD,AI,PSDrot,PSDrotD100,PSDrotD50,PSDH1,PSDH2,PSDeff
    optional:
        freqs_smooth,PSDrot_smooth,PSDrotD100_smooth,
        PSDrotD50_smooth,PSDH1_smooth,PSDH2_smooth,PSDeff_smooth
        
    '''
    import numpy as np
    nn=int(nn)
    dt = 1/fs
    
    n1 = np.size(s1); n2 = np.size(s2); n = np.min((n1,n2))
    s1 = s1[:n]; s2 = s2[:n]   # ensures both components are of the same length
    
    theta = np.arange(0,180)
    thetarad = theta*np.pi/180
    ntheta = len(thetarad)

    if nFFT=='nextpow2':
        nFFT = int(2**np.ceil(np.log2(n)))
    elif nFFT=='same':
        nFFT = int(n)
    elif not isinstance(nFFT, int):
        print('*** error defining nFFT in PSDRotD function ***')
        return
        
    freqs = np.fft.rfftfreq(nFFT, d=dt)
    nfrs = len(freqs)

    if isinstance(dsamp,(float,int)):
        nfrs_smooth=nfrs
    elif isinstance(dsamp,(list,tuple,np.ndarray)):
        nfrs_smooth=len(dsamp)
    else:
        print('*** error defining dsamp in PSDRotD function ***')
        return
    
    PSDrot = np.zeros((ntheta,nfrs))
    PSDrot_smooth = np.zeros((ntheta,nfrs_smooth))
    SD = np.zeros(ntheta)
    AI = np.zeros(ntheta)
    
    for k in range(ntheta):
        sr = s1*np.cos(thetarad[k]) + s2*np.sin(thetarad[k])
        if smo!=0:
            freqs,PSDrot[k,:],SD[k],AI[k],freqs_smooth,PSDrot_smooth[k,:] = PSDFFTEq(sr,fs,duration=duration,nFFT=nFFT,
                                                                basefornFFT = 0,detrend='linear',smo=smo,b=b,dsamp=dsamp)
        else:
            freqs,PSDrot[k,:],SD[k],AI[k] = PSDFFTEq(sr,fs,duration=duration,nFFT=nFFT,
                                                                basefornFFT = 0,detrend='linear',smo=smo,b=b,dsamp=dsamp)
    
    
    PSDrotDnn  = np.percentile(PSDrot,nn,axis=0,method='nearest') 
    PSDH1 = PSDrot[theta==0]
    PSDH2 = PSDrot[theta==90]
    PSDeff = np.sqrt(0.5 *(PSDH1**2 + PSDH2**2))
    
    if smo==0:
        return freqs,SD,AI,PSDrot,PSDrotDnn,PSDH1,PSDH2,PSDeff
    
    
    PSDrotDnn_smooth  = np.percentile(PSDrot_smooth,nn,axis=0,method='nearest') 
    PSDH1_smooth = PSDrot_smooth[theta==0]
    PSDH2_smooth = PSDrot_smooth[theta==90]
    PSDeff_smooth = np.sqrt(0.5 *(PSDH1_smooth**2 + PSDH2_smooth**2))
    
    return (freqs,SD,AI,PSDrot,PSDrotDnn,PSDH1,PSDH2,PSDeff,
            freqs_smooth,PSDrot_smooth,PSDrotDnn_smooth,
            PSDH1_smooth,PSDH2_smooth,PSDeff_smooth)

def FASRotD(s1,s2,fs,nn,nFFT='nextpow2',smo=2,b=20,dsamp=0):
    '''
    Parameters
    ----------
    s1 : 1D array
        component 1 - acceleration time-series
    s2 : 1D array
        component 2 - acceleration time-series
    fs : integer
        sampling frequency
    nn : percentile, e.g. 50 for median, 100 for envelope
    
    nFFT : Optional (defaults to 'nextpow2'), number of points to claculate the FFT, options :
        
        'nextpow2': zero padding until the mext power of 2 is reached
        
        'same': keep the number of points equal to the number of poitns in
                the signal
                
        An integer:  
            If n is smaller than the length of the input, the input is cropped. 
            If it is larger, the input is padded with zeros. 
    
    smo: (integer, defaults to 1):
        0: no smoothing
        1: smooth using Konno-Ohmachi
        2: smooth using moving window of varibale width (NRC SRP3.7.1)
        
        
    b: (float) Used only if smo is 1 or 2:
       
       for smo1:
       coefficient for band width, common values are 188.5 (e.g. Kottke etal 2021)
       - provides a smoothing # operator with a bandwidth of 1/30 of a decade, and
       40 (e.g. Bora et al 2019) which results on a smoother spectrum
       
       for smo2: NRC SRP 3.7.1 reccomends 20 (20% of the central frequency)
    
    dsamp: defaults to 0 : no downsampling
           1D array with the frequencies to dowsample the spectrum
               
    Returns
    -------
    returns:
        freqs, FASrot, FASrotD100, FASrotD50, FASeff, FASH1, FASH2
    optional:
        FASrot_smooth, FASrotD100_smooth, FASrotD50_smooth, FASeff_smooth, FASH1_smooth, FASH2_smooth
        
    '''
    import numpy as np
    import pykooh
    
    dt=1/fs
    n1 = np.size(s1); n2 = np.size(s2); n = np.min((n1,n2))
    s1 = s1[:n]; s2 = s2[:n]   # ensures both components are of the same length
    
    theta = np.arange(0,180)
    thetarad = theta*np.pi/180
    ntheta = len(thetarad)

    if nFFT=='nextpow2':
        nFFT = int(2**np.ceil(np.log2(n)))
    elif nFFT=='same':
        nFFT = int(n)
    elif not isinstance(nFFT, int):
        print('*** error defining nFFT in FASRotD function ***')
        return
    
    freqs = np.fft.rfftfreq(nFFT, d=dt)
    nfrs = np.size(freqs)
    
    FH1  = np.fft.rfft(s1,nFFT)
    FH2  = np.fft.rfft(s2,nFFT)

    MF1,Mthetarad = np.meshgrid(FH1,thetarad,sparse=True, copy=False)
    MF2,_         = np.meshgrid(FH2,thetarad,sparse=True, copy=False)
      
    FASrot = dt*np.abs(MF1*np.cos(Mthetarad)+MF2*np.sin(Mthetarad))

    FASrotDnn  = np.percentile(FASrot,nn,axis=0,method='nearest') 
    FASH1 = FASrot[theta==0]
    FASH2 = FASrot[theta==90]
    FASeff = np.sqrt(0.5 *(FASH1**2 + FASH2**2))
    
    if smo==0:
        return freqs,FASrot,FASrotDnn,FASeff,FASH1,FASH2         
       
    if smo==1:
        FASrot_smooth_allfreq = np.zeros((ntheta,nfrs))
        for k in range(ntheta):
            FASrot_smooth_allfreq[k,:] = pykooh.smooth(freqs, freqs, FASrot[k,:], b)
            
    elif smo==2:
        FASrot_smooth_allfreq = np.copy(FASrot)
        overl = b/100
        if overl>0:
            for k in range(ntheta):
            
                for q in range(1,nfrs-1):
                    lim1 = (1-overl)*freqs[q]
                    lim2 = (1+overl)*freqs[q]
                    
                    if freqs[0]>lim1:
                        lim1 = freqs[0]
                        lim2 = freqs[q]+(freqs[q]-freqs[0])
                    if freqs[-1]<lim2:
                        lim2 = freqs[-1]
                        lim1 = freqs[q]-(freqs[-1]-freqs[q])
                        
                    locsf = np.where((freqs>=lim1)&(freqs<=lim2))
                    FASrot_smooth_allfreq[k,q]=np.mean(FASrot[k,locsf])
    else:
        print('*** error defining smo in FASRotD function ***')
        return
    
    if isinstance(dsamp,(float,int)):
        FASrot_smooth = np.copy(FASrot_smooth_allfreq)
        dsampfreqs=np.copy(freqs)
    elif isinstance(dsamp,(list,tuple,np.ndarray)):
        dsampfreqs=np.array(dsamp)
        nfrs_smooth = len(dsampfreqs)
        FASrot_smooth = np.zeros((ntheta,nfrs_smooth))
        for k in range(ntheta):
            FASrot_smooth[k,:] = log_interp(dsampfreqs, freqs, FASrot_smooth_allfreq[k,:])
    else:
        print('*** error defining dsamp in FASRotD function ***')
        return
    

    FASrotDnn_smooth  = np.percentile(FASrot_smooth,nn,axis=0,method='nearest') 
    FASH1_smooth = FASrot_smooth[theta==0]
    FASH2_smooth = FASrot_smooth[theta==90]
    FASeff_smooth = np.sqrt(0.5 *(FASH1_smooth**2 + FASH2_smooth**2))
    
    return (freqs,FASrot,FASrotDnn,FASeff,FASH1,FASH2,
            dsampfreqs,FASrot_smooth,FASrotDnn_smooth,FASeff_smooth,FASH1_smooth,FASH2_smooth)

def PSDFAScomps2D(nt,dt,envelope1,envelope2,TargetFAS0,TargetFAS90,nn):
    '''
    Generates a synthetic motion comprised of two horizontal 
    components given a target FAS and amplitude envelope for each 
    component. Returns the RoTDnn PSD and of the motion.

    Parameters
    ----------
        
    nt : int
        number of points to generate the motion
    dt : float
        time between samples [s]
    envelope1 : 1D array
        amplitde envelope for H0
    envelope2 : 1D array
        amplitude envelope for H90
    TargetFAS0 : 1D array
        target FAS for H0
    TargetFAS90 : 1D array
        target FAS for H90
    nn : int
        percentile for RotDnn
        
    Returns
    -------
    SD,AI,PSDrotDnn,FASrotDnn
    '''
    
    import numpy as np
    fs= int(1/dt)

    so1 = np.random.randn(nt) 
    so2 = np.random.randn(nt) # generates the synthetic signals
    
    so1 = envelope1*so1
    so2 = envelope2*so2
    
    FSo1 = np.fft.rfft(so1)  # initial Fourier coefficients
    FSo2 = np.fft.rfft(so2)
    
    FASo1 = dt*np.abs(FSo1)  # initial FAS
    FASo2 = dt*np.abs(FSo2) 
    
    ff1 = TargetFAS0 / FASo1 # modification factors
    ff2 = TargetFAS90 / FASo2
    
    FS1 = ff1 * FSo1 # modified Fourier coefficients
    FS2 = ff2 * FSo2
    
    s1 = np.fft.irfft(FS1) # FAS compatible signal
    s2 = np.fft.irfft(FS2)
    
    _,SD,AI,_,PSDrotDnn,_,_,_=PSDRotD(s1,s2,fs,nn,duration=(5,75),nFFT='same',smo=0)
    
    
    _,FASrot,FASrotDnn,_,_,_ = FASRotD(s1,s2,fs,nn,nFFT='same',smo=0)
    
    return SD,AI,PSDrotDnn,FASrotDnn

def WindowAverage(FAS,PSD,freqs,freqs_des,overlap=20):
    import numpy as np
    averagedFAS = np.copy(FAS)
    averagedPSD = np.copy(PSD)
    overl = overlap/100
    nfrs=len(freqs)
    
    if overl>0:
        for k in range(1,nfrs-1):
            lim1 = (1-overl)*freqs[k]
            lim2 = (1+overl)*freqs[k]
            
            if freqs[0]>lim1:
                lim1 = freqs[0]
                lim2 = freqs[k]+(freqs[k]-freqs[0])
            if freqs[-1]<lim2:
                lim2 = freqs[-1]
                lim1 = freqs[k]-(freqs[-1]-freqs[k])
                
            locsf = np.where((freqs>=lim1)&(freqs<=lim2))
            averagedFAS[k]=np.mean(FAS[locsf])
            averagedPSD[k]=np.mean(PSD[locsf])
    
    averagedFAS_interp=log_interp(freqs_des,freqs[1:],averagedFAS[1:])
    averagedPSD_interp=log_interp(freqs_des,freqs[1:],averagedPSD[1:])
    
    
    return averagedFAS_interp,averagedPSD_interp

def RotDnnTargetFASPSD(f_or,ds_or,sd575,freqs_des,workname='RotDnnTargetFASPSD',
                       nnPSA=100,nnPSD=100,sdratio=1.3, 
                       smo=2,b=20,
                       zi=0.05,F1=0.2,F2=50,
                       allow_err=2.5,neqsPSD=1000,plots=1):
    '''
    

    Parameters
    ----------
    f_or : 1D array - [Hz] frequencies where the target response spectrrum is defined
    
    ds_or : 1D array - [g] target spectral accelerations
    
    sd575 : float - [s] expected strong motion duration sd5-75
    
    freqs_des : 1D array -[Hz] # frequencies where the output target FAS and PSD would be interpolated
    
    workname : sting - name for output files. The default is 'RotDnnTargetFASPSD'
    
    nnPSA : int - percentile for target PSA. The default is 100
    
    nnPSD : int - percentile for FAS and PSD. The default is 100
    
    sdratio : float - ratio between the hor. components sd5-75. The default is 1.3
    
    smo: (integer, defaults to 2):
        0: no smoothing
        1: smooth using Konno-Ohmachi
        2: smooth using moving window of varibale width (NRC SRP3.7.1)
        
    b: (float, defaults to 20) Used only if smo is 1 or 2:
        
       for smo1: coefficient for band width, common values are 188.5 (e.g. Kottke etal 2021)
       - provides a smoothing operator with a bandwidth of 1/30 of a decade, and
       40 or 20 (e.g. Bora et al 2019) which results on a smoother spectrum
       
       for smo2: NRC SRP 3.7.1 reccomends 20 (20% of the central frequency)
    
    zi : float - damping ratio for PSA. The default is 0.05
    
    F1, F2 : floats - frequencies defining the range to check PSA compatibility
                      Default to 0.2 and 50
                      
    allow_err : float - acceptble mismatch in PSA. The default is 2.5
    
    neqsPSD : int - number of motions used to generated target FAS and PSD. The default is 1000
    
    plots : integer - 1 generate plots. The default is 1

    Returns
    -------
    averagedFAS : 1D array - target FAS
    averagedPSD : 1D array - target PSD

    '''
    
    
    import numpy as np 
    import concurrent.futures
    import pykooh
    
    workname = str(workname)
    
    g = 9.81
    fs = 200     # sampling frequency [Hz]
    dt = 1/fs
    ds_or = ds_or*g
    
    # Create time envelopes:  
    
    if sdratio<1:
        sdratio=1/sdratio
        
    sd90 = (sd575**2/sdratio)**0.5 # different sd for each component
    sd0 = sdratio*sd90 # assumes sd given is the geomean
    
    # envelope 0:
        
    tf0 = 3.54*sd0  # total duration of the signal
    
    nt0 = int(tf0/dt)+1
    if nt0%2!=0: tf0+=dt; nt0+=1 # Adjust time vector for even length
    
    t0 = np.linspace(0,tf0,nt0)
    
    envelope0 = saragoni_hart_w(nt0,eps=0.2,n=0.2,tn=0.6)
    
    # envelope 90:
        
    tf90 = 3.54*sd90  # total duration of the signal
    
    nt90 = int(tf90/dt)+1
    
    t90 = np.linspace(0,tf90,nt90)
    
    envelope90 = saragoni_hart_w(nt90,eps=0.2,n=0.2,tn=0.6)
    
    # ensure both envelopes are of the same lenght:

    envelope90 = np.hstack((envelope90,0))
    t90 = np.hstack((t90,t0[-1]))
    envelope90 = np.interp(t0,t90,envelope90)
    
    nt = nt0
    
    
    #######################################
    
    sets = [5,10,20,30,40,50,60,70,80,90,100] 
    nsets = np.size(sets)
    freqs = np.fft.rfftfreq(nt, d=dt) # Fourier frequencies

    # Frequencies where the response spectra will be compute (using all 
    # Fourier frequencies is to expensive):
    f = np.hstack((np.array([0.04,0.06,0.08]),np.geomspace(0.1,50,100),
                   np.array([55,60,70,80,90,100])))
    
    # Redefine f so that all response spectra frequencies are within the 
    # Fourier frequencies:

    if f[0]>freqs[1]:
        f[0] = freqs[1]

    if f[-1]<freqs[-1]:
        f[-1] = freqs[-1]
 
    f = f[(f>=0.999*freqs[1])&(f<=1.001*freqs[-1])]
    
    # Check the frequency range where the given target spectrum was defined:
    
    if f_or[0] > freqs[1]:
        raise Exception(f'''the target response spectrum is currently defined 
                        from {f_or[0]:.4f} Hz but needs to be defined at least 
                        from {0.98*freqs[1]:.4f} Hz''')
    
    if np.isclose(f_or[-1],freqs[-1]):
        freqs[-1]=f_or[-1]
        
    if f_or[-1] < freqs[-1]:
        raise Exception(f''''the target response spectrum is currently defined 
                        until {f_or[-1]:.4f} Hz but needs to be defined at 
                        least until {freqs[-1]:.4f} Hz''')
    
    # Check the frequency range [F1,F2] specified to check the match:                
    
    if F1<f[0]:
        raise Exception(f'''error defining the matching range [F1 F2], 
                        F1 shall be > {f[0]:.4f}Hz''')

    if F2>f[-1]:
        raise Exception(f'''error defining the matching range [F1 F2], 
              F2 shall be < {f[-1]:.4f}Hz''')
    
    ds =  log_interp(f,f_or,ds_or)        # log-interpolate the desig spectrum
                                          # to the frequencies where the response
                                          # spectra would be computed
                                          
    locs = np.where((f>=F1)&(f<=F2))[0]   # positions within frequency range for match check
    
    #initial FAS for each component H0 and H90:
    
    ds_freqs = log_interp(freqs[1:], f, ds)    # Interpolate target spectrum 
                                               # at the fourier frequencies
    ds_freqs = np.concatenate(([0], ds_freqs))
    
    ratio0 = FASPSAratio(freqs[1:],sd0)  # FAS/PSA ratio Montejo and Vidot 2017
                                     # while not develpoed for RotDnn is just a first try
                                     # to have a starting point               
    TFAS0 = ds_freqs[1:]*ratio0          # target FAS for H0
    TFAS0 = np.concatenate(([0], TFAS0))
    
    ratio90 = FASPSAratio(freqs[1:],sd90)                        
    TFAS90 = ds_freqs[1:]*ratio90          # target FAS for H90
    TFAS90 = np.concatenate(([0], TFAS90))
    
    T = 1/f
    
    
    ################################
    
    PSAavg = np.zeros((len(T),nsets))            # stores the average PSA per set
    
    FAS0target= np.zeros((len(freqs),nsets))     # stores individual target compoenent FAS before each iteration
    FAS90target = np.zeros((len(freqs),nsets))
    
    FAS0target[:,0] = TFAS0
    FAS90target[:,0] = TFAS90
    
    calc_errs = np.zeros(nsets)
    
    print('*'*20)
    print('Now generating spectrum compatible FAS')
    print(f'Target error: {allow_err:.2f}%, max # of iters.: {nsets}')
    print('*'*20)
    
    for k in range(nsets):
        
        PSA  = np.zeros((len(T),sets[k]))        # stores individual record PSA, used to get the average
                                                 # after the loops are completed, reset each k loop
                                                  
        with concurrent.futures.ProcessPoolExecutor() as executor:
            PSAallrecords = [executor.submit(FASPSAcomps2D,*[nt,dt,envelope0,envelope90,FAS0target[:,k],FAS90target[:,k],T,zi,nnPSA]) for _ in range(sets[k])]
        
        q=0      
        for PSAsinglerecord in PSAallrecords:
            PSA[:,q]=PSAsinglerecord.result()
            q=q+1
    
        PSAavg[:,k] = np.mean(PSA,axis=1) # takes average PSA per set  
        diflimits = np.abs(ds[locs]-PSAavg[locs,k])/ds[locs]
        calc_errs[k] = np.mean(diflimits)*100
         
        print(f'iteration #: {k+1} - set with {sets[k]} records - error: {calc_errs[k]:.2f}%')
          
        if calc_errs[k]<allow_err:
            PSAavg = PSAavg[:,:k+1]
            FAS0target = FAS0target[:,:k+1]
            FAS90target = FAS90target[:,:k+1]
            calc_errs = calc_errs[:k+1]
            print(f'error satisfied at iteration # {k+1} - error: {calc_errs[k]:.2f}%')
            break
        elif k!=nsets-1:
            PSAavg_interp = log_interp(freqs[1:], f, PSAavg[:,k]) # iterpolates to fourier frequencies 
                                                                  # to allow ratios calculation
            factor = ds_freqs[1:]/PSAavg_interp                   # take ratios between target and 
                                                                  # response spectra
            FAS0target[1:,k+1]=factor*FAS0target[1:,k]            # apply ratios to get updated target PSD
            FAS90target[1:,k+1]=factor*FAS90target[1:,k]
    else:
        
        print('max number of iterations was reached, error was not satisfied')
        print('the results from the iteration with the lowest error would be used')
   
    nsets = np.size(calc_errs)
    aux = np.arange(1,nsets+1)
    
    minerrloc = np.argmin(calc_errs)
    TFAS0fin = FAS0target[:,minerrloc]
    TFAS90fin = FAS90target[:,minerrloc]
    PSAfin = PSAavg[:,minerrloc] 
    
    SDfin = np.zeros((180,neqsPSD))
    AIfin = np.zeros((180,neqsPSD))
    PSDrotDnnfin = np.zeros((len(freqs),neqsPSD))
    FASrotDnnfin = np.zeros((len(freqs),neqsPSD))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        allrecords = [executor.submit(PSDFAScomps2D,*[nt,dt,envelope0,envelope90,TFAS0fin,TFAS90fin,nnPSD]) for _ in range(neqsPSD)]
    
    q=0
    for singlerecord in allrecords:
        SDfin[:,q] = singlerecord.result()[0]
        AIfin[:,q] = singlerecord.result()[1]
        PSDrotDnnfin[:,q] = singlerecord.result()[2]
        FASrotDnnfin[:,q] = singlerecord.result()[3]
        q=q+1
        
    theta = np.arange(0,180)
    geomSD = np.mean(np.sqrt(SDfin[0]*SDfin[90]))
    
    PSDrotDnnfin_mean=np.mean(PSDrotDnnfin,axis=1)
    FASrotDnnfin_mean=np.mean(FASrotDnnfin,axis=1)
    
    if smo==2:
        averagedFAS,averagedPSD=WindowAverage(FASrotDnnfin_mean,PSDrotDnnfin_mean,freqs,freqs_des,overlap=b)
    elif smo==1:
        averagedFAS=pykooh.smooth(freqs, freqs, FASrotDnnfin_mean, b)
        averagedFAS=log_interp(freqs_des, freqs, averagedFAS)
        averagedPSD=pykooh.smooth(freqs, freqs, PSDrotDnnfin_mean, b)
        averagedPSD=log_interp(freqs_des, freqs, averagedPSD)
    else: # no smooth
        print('tartget FAS and PSD spectra are not smoothed')
        averagedFAS=log_interp(freqs_des, freqs, FASrotDnnfin_mean)
        averagedPSD=log_interp(freqs_des, freqs, PSDrotDnnfin_mean)
    alltog = np.column_stack((freqs_des,averagedFAS,averagedPSD)) 
    np.savetxt(workname+'_targetFASandPSD.txt',alltog,fmt='%.8f',header='freqs.[Hz] - FAS [m/s] - PSD [m2/s3]')
    if plots:
        
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        mpl.rcParams['font.size'] = 9 
        mpl.rcParams['legend.frameon'] = False
        mpl.rcParams['mathtext.fontset'] = 'dejavuserif'
              
        plt.figure(figsize=(7,4))
        
        plt.subplot(131)
        plt.plot(aux,calc_errs,'--o',color='black',ms=4,lw=1,mfc='pink')
        plt.xlabel('iteration #')
        plt.ylabel('error [%]')
        
        plt.subplot(132)
        plt.semilogx(f,ds/g,color='black',lw=1,label='target')
        plt.semilogx(f,PSAfin/g,'-',color='blueviolet',lw=1,label='set mean')
        plt.xlim((0.08,100))
        plt.ylabel('PSA [g]')
        plt.xlabel('F [Hz]')
        plt.legend(handlelength=1)
        
        plt.subplot(133)
        plt.semilogx(freqs,TFAS0fin,color='cornflowerblue',lw=1,label='H1')
        plt.semilogx(freqs,TFAS90fin,color='salmon',lw=1,label='H2')
        plt.xlim((0.08,100))
        plt.ylabel('Target component FAS [m/s]')
        plt.xlabel('F [Hz]')
        plt.legend()
        
        plt.tight_layout(h_pad=0,w_pad=0)
        plt.savefig(workname+' Target Component FAS.jpg',dpi=300,bbox_inches='tight')
        
        plt.figure(figsize=(7,4))
        
        plt.subplot(121)
        plt.loglog(1,1,color='silver',lw=0.5,label='ind.motions')
        plt.loglog(freqs,FASrotDnnfin,color='silver',lw=0.5)
        plt.loglog(freqs,FASrotDnnfin_mean,color='black',lw=1,label='mean')
        plt.loglog(freqs_des,averagedFAS,'-',color='salmon',lw=1,mfc='white',ms=3,label='smooth')
        plt.xlim((0.08,100))
        plt.ylim((FASrotDnnfin_mean[-1],np.max(FASrotDnnfin)))
        plt.xlabel('F [Hz]')
        plt.ylabel('FASRotDnn [m/s]')
        plt.legend()
        
        plt.subplot(122)
        plt.loglog(freqs,PSDrotDnnfin,color='silver',lw=0.5)
        plt.loglog(freqs,PSDrotDnnfin_mean,color='black',lw=1)
        plt.loglog(freqs_des,averagedPSD,'-',color='salmon',lw=1,mfc='white',ms=3)
        plt.xlim((0.08,100))
        plt.ylim((PSDrotDnnfin_mean[-1],np.max(PSDrotDnnfin)))
        plt.xlabel('F [Hz]')
        plt.ylabel(r'PSDRotDnn $[m^2/s^3]$')
        
        plt.tight_layout(h_pad=0,w_pad=0)
        plt.savefig(workname + ' Target RotDnn FAS and PSD.jpg',dpi=300,bbox_inches='tight')
        
        plt.figure(figsize=(6,3))
        plt.plot(theta,SDfin,color='silver',lw=0.5)
        plt.hlines(sd575,theta[0],theta[-1],color='salmon',label='target')
        plt.hlines(geomSD,theta[0],theta[-1],linestyles='dashed',color='black',label='average geom. mean')
        
        plt.legend()
        plt.xticks([0,45,90,135,180])
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$SD_{5-75}$ [s]')
        plt.tight_layout(h_pad=0,w_pad=0)
        plt.savefig(workname + 'Duration verification.jpg',dpi=300,bbox_inches='tight')
        
    return averagedFAS,averagedPSD


    




    