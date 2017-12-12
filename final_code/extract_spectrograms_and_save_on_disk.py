#------------------------------------------------------------------------    
# Load userdefined modules
import audio
import dataset
import model
from dataset import prepare_data
from dataset import load_data
from dataset import compute_global_norm
from helper import makeDirectory

def make_data():
    fs=16000
    fft_size=2048  #256  # 512
    win_size=2048  #256  #512
    hop_size=160

    duration=4
    targets=2
    inputType='mag_spec'
    
    spectrogramPath='/homes/bc305/myphd/stage2/deeplearning.experiment1/spectrograms/'
        
    basePath='/import/c4dm-datasets/SpeakerRecognitionDatasets/ASVSpoof2017/'
    outPath = spectrogramPath+ inputType + '/'+str(fft_size)+ 'FFT/' + str(duration)+ 'sec/'
                
    # Prepare training data
    print('Preparing the training data')
    prepare_data(basePath,'train',outPath,inputType,duration,targets,fs,fft_size,win_size,hop_size)

    # Prepare Validation data
    print('Preparing the validation data')
    prepare_data(basePath,'dev',outPath,inputType,duration,targets,fs,fft_size,win_size,hop_size)
    
    # Prepare test data
    print('Preparing the test data')
    prepare_data(basePath,'test',outPath,inputType,duration,targets,fs,fft_size,win_size,hop_size)

    
make_data()
