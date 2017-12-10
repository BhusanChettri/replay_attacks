#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.io.wavfile as wav
import librosa
import numpy as np

#import librosa.display  #for display
#import matplotlib.pyplot as plt

#########################################################################################################################

def compute_mean_std(arrays, axis=0, keepdims=False, dtype=np.float64, ddof=1):
    """
    Computes the total mean and standard deviation of a set of arrays across
    a given axis or set of axes, using Welford's algorithm.
    
    This function is taken from Jan Schlüter znorm.py
    """
    kwargs = dict(axis=axis, keepdims=True, dtype=dtype)
    n = m = s = 0
            
    for data in arrays:
        n += len(data)
        delta = (data - m)
        m += delta.sum(**kwargs) / n
        s += (delta * (data - m)).sum(**kwargs)
    s /= (n - ddof)
    np.sqrt(s, s)
    
    if not keepdims:
        axes = (axis,) if isinstance(axis, int) else axis
        index = tuple(0 if a in axes else slice(None) for a in range(m.ndim))
        return m[index], s[index]
    
    return m, s

#########################################################################################################################

def update_audio_samples(fs, samples, threshold, remove=0.1, removeFlag=False):
    '''
    Inputs:
    audioFile : the absolute path of the audio file
    threshold : is in seconds. Used to trim and append audio samples
    remove    : is in seconds. audio file to remove from start.Default is 100ms that is equal to 1600 raw samples
                and avoid computational issues.                
    Output:
    trimmed/appended audio file of length given by threshold
    
    Threshold is in seconds. So convert it into samples first.
    If file is large, we throw away samples after the threshold else we copy the samples to match threshold
    '''
        
    #fs, samples = wav.read(audioFile)
    # We do this using librosa in the calling function now
    
    #If removeFlag is set then remove samples from start
    if removeFlag:
        remove = int(remove * fs)         
        samples = samples[remove:]                                          
    
    threshold_samples = threshold * fs      
    audio_length = len(samples)/fs
            
    if audio_length < threshold:   #replicate the samples to match threshold        
        n=0
        while n<threshold_samples:            
            samples = np.tile(samples, 3) #appends 3 copies of samples            
            n+=len(samples)            
        samples = samples[0:threshold_samples] #just take threshold_samples                   
        
    elif audio_length > threshold:        
        samples = samples[0:threshold_samples] #just take threshold_samples and ignore rest
        
    #print('New length after appending/truncating = %s seconds' % (len(samples)/fs))
    
    return samples

#########################################################################################################################

def minMaxNormalize(data, columnAxisNormalise=False):
    '''
    data is a matrix of size [RxC] where R is the number of rows and C the columns
    If columnAxisNormalise is set to true then the code finds max and min along the columns and normalizes
    the corresponding column entries to 0-1. But it seems to make no sense. If we want to scale our whole 
    data to 0-1 then better to not do it column wise.
    '''
    
    # This is incorrect. 
    # TODO : need to do column wise across time
    if columnAxisNormalise:
        mx=np.max(data, axis=0)
        mn=np.min(data, axis=0)        
    else:
        mx=np.max(data)
        mn=np.min(data)
        
    return (data-mn)/(mx-mn)

#########################################################################################################################

def zNormalizeData(data, columnAxisNormalise=False):
    '''
    Perform zero mean and unit variance normalization.
    Remove mean from each data point and divide by standard deviation. As a result, the data would lose the origin and 
    scale information as it will all have zero mean and unit variance.
    '''
    # this is incorrect.
    # at the moment using Jan Schlüter znorm.py
    
    if columnAxisNormalise:
        mn = np.mean(data, axis=0)
        std = np.std(data, axis=0)        
    else:
        mn = np.mean(data)
        std = np.std(data)
        
    return (data-mn)/std    

#########################################################################################################################
def compute_spectrogram(input_type, filename, fft_size=512, win_size=512, hop_size=160, duration=1):
        
        
    samples, sr = librosa.load(filename, sr=None, dtype=np.float32)    
        
    #Truncate or append samples based on duration
    samples = update_audio_samples(sr,samples,duration)
       
    #Take the FFT
    D = librosa.stft(samples,fft_size,hop_size,win_size)    
    
    if input_type == 'mag_spec': #power magnitude spectrogram        
        D= np.log(np.maximum((np.abs(D)**2), 1e-7))
    elif input_type == 'mel_spec':
        print('to do for mel spectrogram code')
    elif input_type == 'cqt_spec':
        print('to do for cqt spectrogram code')
                    
    r,c = D.shape
    
    #Let us return the spectrogram matrix in timeXfrequency format by taking transpose
    return np.transpose(D[:, 0:c-1])


def compute_spectrogramOLD(filename, fft_size=512, win_size=512, hop_size=160, duration=1):
        
    """
    Input parameters:
    filename  = audio file to compute normalized log power spectrogram    
    fft_size  = fft window size is 2048 by default as we want to give emphasis on frequency.
    win_size  = fft_size
    hop_size  = 10ms that corresponds to 160 samples
    duration  = 3second by default.   
    
    Output:
    Log normalized power magnitude spectrogram of the audio "filename"
    
    Defaults:
    fs       = 16000  #16khz
    fft_size = 2048   #128ms 
    win_size = 2048   #128ms
    hop_size = 160    #10ms
    duration = 3      #3000ms    
    """          
    
    #Truncate or append samples based on duration
    samples = update_audio_samples(filename, duration)
               
    #Take the FFT
    D = librosa.stft(samples, n_fft=fft_size, hop_length=hop_size, win_length=win_size)
            
    #Take the power of the magnitude spectrum. Just take the square.
    D = np.square(np.abs(D))
    
    #Note that many audio files contains many zeros in initial samples which causes issues 
    #Instead of removing first few samples that are mostly 0.
    #Replace all 0's of File with a small value 1e-10 in pythonic way  
    
    #D[D == 0] = 1   # Replacing all 0's by 1 will ultimately make it zero when we take log
                     # because log(1)=0
    
    ## Changed as per suggestion by Saumitra
    print('Replacing tiny number and zeros by 1e-7')
    D = np.maximum(D,1e-7)
    
    #Take the log
    try:
        D = np.log(D)
    except e:
        print('Cannot take np.log for File %s ' %(filename))
        
    r,c = D.shape
    
    #Let us return the spectrogram matrix in timeXfrequency format by taking transpose    
    return np.transpose(D[:, 0:c-1])

'''
The function used by Russians for computing spectrogram

function Y=Spectr(s,W,SP)
    wnd=hamming(W);
    y=enframe(s,wnd,fix(W*SP))';  #fix(W*SP) is the actual hop they use for framing not 100ms
    Y=fft(y);  
    z=fix(size(Y,1)/2);
    Y=abs(Y(1:z+1,:));
end

where
s  = signal (vector)
W  = window length (integer, 1728 in our implementation) 
SP = relative hop size (float, 0.1 in our implementation). 
enframe is the function from VoiceBox library.

We will have to double-check later !
'''

#########################################################################################################################

def trimFrequency(spec, low_cut_off, high_cut_off, no_of_bins):
    '''
    Inputs
         spec      : normalized input spectrogram of dimension [TXF] where T corresponds to frames (time)
                     and F  the frequency bins                     
         no_of_bins: the number of frequency bins to be removed from spectrogram.
                     It removes it starting from 1st frequency bin.
         
    '''
    return spec[:, no_of_bins:]

#########################################################################################################################

def plotSpec(spec):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)    
    librosa.display.specshow(spec, y_axis='log', x_axis='time')
    plt.title('Log-frequency normalized power spectrogram')
    plt.show()
    
