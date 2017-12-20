#------------------------------------------------------------------------    
# Load userdefined modules
import audio
import dataset
import model
from dataset import prepare_data
from dataset import load_data
from dataset import compute_global_norm
from helper import makeDirectory

def make_data_mag_spectrogram():
    
    fs=16000
    fft_size=512   #256  # 512
    win_size=512   #256  #512
    hop_size=160

    duration=1    
    inputType='mag_spec'
    
    augment=True #True
    data_window=100   # for FFT based and for cqt = 
    window_shift=100   #each frame is 32ms, 10 window shift corresponds to 320ms
    save=True
    #minimum_length=1  # in seconds
    
    if augment:
        spectrogramPath='/homes/bc305/myphd/stage2/deeplearning.experiment1/spectrograms_augmented/1sec_shift/'
    else:
        spectrogramPath='/homes/bc305/myphd/stage2/deeplearning.experiment1/spectrograms/'        
        
    basePath='/import/c4dm-datasets/SpeakerRecognitionDatasets/ASVSpoof2017/'
    outPath = spectrogramPath+ inputType + '/'+str(fft_size)+ 'FFT/' + str(duration)+ 'sec/'
                
    # Prepare training data
    #print('Preparing the training data')
    #prepare_data(basePath,'train',outPath,inputType,duration,fs,fft_size,win_size,hop_size,data_window,window_shift,
    #             augment,save)

    # Prepare Validation data
    #print('Preparing the validation data')
    #prepare_data(basePath,'dev',outPath,inputType,duration,fs,fft_size,win_size,hop_size,data_window,window_shift,
    #             augment,save)
    
    # Prepare test data
    print('Preparing the test data')
    prepare_data(basePath,'test',outPath,inputType,duration,fs,fft_size,win_size,hop_size,data_window,window_shift,
    augment,save)
    
def make_data_mel_spectrogram():
    fs=16000
    fft_size=512
    win_size=512
    hop_size=160

    duration=1
    inputType='mel_spec'
    augment=True
    data_window=100    # for FFT based and for cqt =    
    window_shift=100   #each frame is 32ms, 10 window shift corresponds to 320ms
    save=True
    #minimum_length=1  #in seconds
    
    if augment:
        spectrogramPath='/homes/bc305/myphd/stage2/deeplearning.experiment1/spectrograms_augmented/1sec_shift/'
    else:
        spectrogramPath='/homes/bc305/myphd/stage2/deeplearning.experiment1/spectrograms/'            
        
    basePath='/import/c4dm-datasets/SpeakerRecognitionDatasets/ASVSpoof2017/'
    outPath = spectrogramPath+ inputType + '/'+str(fft_size)+ 'FFT/' + str(duration)+ 'sec/'
                
    # Prepare training data
    #print('Preparing the training data')
    #prepare_data(basePath,'train',outPath,inputType,duration,fs,fft_size,win_size,hop_size,data_window,window_shift,
    #             augment,save)

    # Prepare Validation data
    #print('Preparing the validation data')
    #prepare_data(basePath,'dev',outPath,inputType,duration,fs,fft_size,win_size,hop_size,data_window,window_shift,
    #             augment,save)
    
    # Prepare test data
    print('Preparing the test data')
    prepare_data(basePath,'test',outPath,inputType,duration,fs,fft_size,win_size,hop_size,data_window,window_shift,
    augment,save)    

def make_data_cqt_spectrogram():
    # Note that in audio.py, the spectrogram for CQT uses default parameters. Later we may want to think over this !
    # Thus these parameters being passed has no effect. 
    
    fs=16000
    fft_size=512
    win_size=512
    hop_size=160

    duration=1
    
    inputType='cqt_spec'  #1second correspond to (32, 84) spectrogram in current default configurations
    augment=True
    data_window=32    
    window_shift=30   # keep 30 frames as shift (making 32 will discard lot of frames)  
    save=True
    
    if augment:
        spectrogramPath='/homes/bc305/myphd/stage2/deeplearning.experiment1/spectrograms_augmented/1sec_shift/'
    else:
        spectrogramPath='/homes/bc305/myphd/stage2/deeplearning.experiment1/spectrograms/'        
    
    
    basePath='/import/c4dm-datasets/SpeakerRecognitionDatasets/ASVSpoof2017/'
    outPath = spectrogramPath+ inputType + '/'+str(fft_size)+ 'FFT/' + str(duration)+ 'sec/'
                
    # Prepare training data
    #print('Preparing the training data')
    #prepare_data(basePath,'train',outPath,inputType,duration,fs,fft_size,win_size,hop_size,data_window,window_shift,
    #             augment,save)

    # Prepare Validation data
    #print('Preparing the validation data')
    #prepare_data(basePath,'dev',outPath,inputType,duration,fs,fft_size,win_size,hop_size,data_window,window_shift,
    #             augment,save)
    
    # Prepare test data
    print('Preparing the test data')
    prepare_data(basePath,'test',outPath,inputType,duration,fs,fft_size,win_size,hop_size,data_window,window_shift,
    augment,save)        
    
make_data_mag_spectrogram()
make_data_mel_spectrogram()
make_data_cqt_spectrogram()

