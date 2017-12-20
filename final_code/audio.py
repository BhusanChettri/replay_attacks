#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.io.wavfile as wav
import librosa
import numpy as np


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

def compute_spectrogram(input_type,filename,fft_size=512, win_size=512, hop_size=160, duration=1,
                        data_augment=True,minimum_length=1):
    
   
    samples, fs = librosa.load(filename, sr=None, dtype=np.float32)    
    
    if data_augment:       
        audio_length = len(samples)/fs   
        #print('Audio length = ', audio_length)
        
        if audio_length < 1.0 :
            minimum_length=1.0
            
        elif audio_length > 1.0 and audio_length < 2.0:   #between 1-2 seconds
            minimum_length=2.0
            
        elif audio_length > 2.0 and audio_length < 3.0:   #between 2-3 seconds
            minimum_length=3.0
            
        elif audio_length > 3.0 and audio_length < 4.0:   #between 3-4 seconds
            minimum_length=4.0
            
        elif audio_length > 4.0 and audio_length < 5.0:   #between 4-5 seconds
            minimum_length=5.0
            
        elif audio_length > 5.0 and audio_length < 6.0:   #between 4-5 seconds
            minimum_length=6.0
            
        elif audio_length > 6.0 and audio_length < 7.0:   #between 6-7 seconds
            minimum_length=7.0
            
        elif audio_length > 7.0 and audio_length < 8.0:   #between 7-8 seconds
            minimum_length=8.0
            
        elif audio_length > 8.0 and audio_length < 9.0:   #between 8-9 seconds
            minimum_length=9.0
            
        elif audio_length > 9.0 and audio_length < 10.0:   #between 9-10 seconds
            minimum_length=10.0
            
        elif audio_length > 10.0 and audio_length < 11.0:   #between 10-11 seconds
            minimum_length=11.0        
        
        #print('Original length and new length = %.2f,%.2f' %(audio_length,minimum_length))
        samples = update_audio_samples(fs,samples,minimum_length)
            
    else:   
        # The old approach. Just truncate or append samples based on duration supplied.
        samples = update_audio_samples(fs,samples,duration)            
    
    
    if input_type == 'cqt_spec':
        #print('take cqt transform')
        
        # Using the default parameters ! need to check this part
        D = np.abs(librosa.cqt(samples,fs))   #, hop_length=160, fmin=20,n_bins=96)    
        D = np.log(np.maximum(D,1e-7))
        
        #print('After CQT, shape = ', D.T.shape)
        
        '''
        #1second correspond to (32, 84) spectrogram in current default configurations
        
        In this case we ensure that each audio file is 3sec long by copying the contents
        Therefore, D = 94x84, 94 is the time and 84 the frequency bins
        As of now we used default parameters of librosa. May have to change it later
        Hop_length is 512 which accounts to 32ms because FS=16000 Hz.
        
        In other words we will always have atleast 94X84 representation even if original audio is less than
        3seconds. From data augmentation context, we can chose 1.5 seconds (half of it) as the orginal size and 
        slide our window every one frame and generate lots of data !  
        So, remember that under CQT when using Data augmentation, we will use 1.5 seconds time, i.e the matrix 
        will be 47x84 (if we use the same default settings)
        '''
                                        
    else:       
        D = librosa.stft(samples,fft_size,hop_size,win_size) 
        
        if input_type == 'mag_spec': #power magnitude spectrogram
            D= np.log(np.maximum((np.abs(D)**2), 1e-7))
            
        elif input_type == 'mel_spec':
            # We compute energy-based mel spectrogram, thus power=1.0 else for power-based mel spectrogram
            # pass power=2.0
            
            D = librosa.feature.melspectrogram(samples, sr=fs,n_fft=fft_size,hop_length=hop_size,
                                                      power=1.0,n_mels=80)
            D = np.log(np.maximum(D,1e-7))
            
    r,c = D.shape
        
    if data_augment:
        return np.transpose(D)
    else:
        return np.transpose(D[:, 0:c-1])   # without data_augmentation we return dropping one frame   
        
    
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
    before = len(samples)/fs
            
    #If removeFlag is set then remove samples from start
    if removeFlag:
        remove = int(remove * fs)         
        samples = samples[remove:]                                          
    
    threshold_samples = int(threshold) * fs      
    audio_length = len(samples)/fs
            
    if audio_length < threshold:   #replicate the samples to match threshold        
        n=0
        while n<threshold_samples:            
            samples = np.tile(samples, 3) #appends 3 copies of samples            
            n+=len(samples)            
        samples = samples[0:threshold_samples] #just take threshold_samples                   
        
    elif audio_length > threshold:        
        samples = samples[0:threshold_samples] #just take threshold_samples and ignore rest
    
    after = len(samples)/fs
    print('Before and after: %.2f,%.2f secs' % (before,after))
    
    return samples    

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
    
