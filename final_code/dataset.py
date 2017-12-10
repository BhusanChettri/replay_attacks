from __future__ import print_function
import numpy as np
import audio
import os
from utility import makeDirectory

#------------------------------------------------------------------------------------------------------------------------------ 

def get_labels_according_to_targets(protocalFile, targets=2):
    # This will not work for Test Data as we do not have labels !
    # Careful using it on test data    
    
    gen= '- - -'
    #------------------ These are spoof conf found in training set
    spf1='E02 P02 R04' 
    spf2='E05 P05 R04'
    spf3='E05 P10 R04'
    #------------------
    
   #Following are spoofing config found in dev set
    spf4='E05 P08 R08'
    spf5='E05 P08 R07'
    spf6='E05 P08 R11'
    spf7='E03 P15 R08'
    spf8='E03 P15 R07'
    spf9='E03 P15 R11'
    spf10='E05 P04 R03'
    spf11='E02 P07 R02'
    spf12='E01 P09 R06'
    spf13=spf1   # this is similar to spf1, so when using train+dev we use 13 configs not 14    
    
    labels=list()
    with open(protocalFile, 'r') as f:
        if targets==2:
            labels = [[1,0] if line.strip().split(' ')[1] == 'genuine' else [0,1] for line in f] #test this
            print(len(labels))
            
        else:
            for line in f:
                units = line.strip().split(' ')
                config = units[4]+' '+units[5]+' '+units[6]

                if targets ==4: # Using only Training set
                    # This is tested. Creates the label correctly on train,dev
                    # Not tested on Eval as it has no labels
                    if config == gen:
                        label=np.array([1,0,0,0]) # Genuine config
                    elif config == spf1 or config==spf11 or config==spf12 or config==spf13: 
                        label=np.array([0,1,0,0]) # Type1 spoof config
                    elif config == spf2 or config==spf4 or config==spf5 or config==spf6 or config==spf10:
                        label=np.array([0,0,1,0]) # Type2 spoof config               
                    elif config == spf3 or config==spf7 or config==spf8 or config==spf9:
                        label=np.array([0,0,0,1]) # Type3 spoof config
                    labels.append(label)
                    
                elif targets == 11:  # using only development set configs
                    # Need to map training data cleverly coz it only has 3 configs. 
                    if config == gen:
                        label=np.array([1,0,0,0,0,0,0,0,0,0,0])                
                    elif config == spf4:
                        label=np.array([0,1,0,0,0,0,0,0,0,0,0])
                    elif config == spf5:
                        label=np.array([0,0,1,0,0,0,0,0,0,0,0]) 
                    elif config == spf6:
                        label=np.array([0,0,0,1,0,0,0,0,0,0,0]) 
                    elif config == spf7 or config == spf3:
                        label=np.array([0,0,0,0,1,0,0,0,0,0,0]) 
                    elif config == spf8:
                        label=np.array([0,0,0,0,0,1,0,0,0,0,0])
                    elif config == spf9:
                        label=np.array([0,0,0,0,0,0,1,0,0,0,0])
                    elif config == spf10 or config == spf2:
                        label=np.array([0,0,0,0,0,0,0,1,0,0,0])
                    elif config == spf11:
                        label=np.array([0,0,0,0,0,0,0,0,1,0,0])
                    elif config == spf12:
                        label=np.array([0,0,0,0,0,0,0,0,0,1,0])
                    elif config == spf13 or config == spf1:
                        label=np.array([0,0,0,0,0,0,0,0,0,0,1])
                    labels.append(label)
                    
                elif targets == 13: # using train+dev
                    # 14 dimensional one-hot vector
                    # Under this category use train data as validation because it has
                    # less number of spoofing instances that is different from dev
                    # This one also tested on train,dev but for eval does not work                    
                    # that is the reason why after training network extract features and train another 
                    # classifier
                    
                    if config == gen:
                        label=np.array([1,0,0,0,0,0,0,0,0,0,0,0,0])             
                    elif config == spf1:
                        label=np.array([0,1,0,0,0,0,0,0,0,0,0,0,0])             
                    elif config == spf2:
                        label=np.array([0,0,1,0,0,0,0,0,0,0,0,0,0])             
                    elif config == spf3:
                        label=np.array([0,0,0,1,0,0,0,0,0,0,0,0,0])             
                    elif config == spf4:
                        label=np.array([0,0,0,0,1,0,0,0,0,0,0,0,0])             
                    elif config == spf5:
                        label=np.array([0,0,0,0,0,1,0,0,0,0,0,0,0])             
                    elif config == spf6:
                        label=np.array([0,0,0,0,0,0,1,0,0,0,0,0,0]) 
                    elif config == spf7:
                        label=np.array([0,0,0,0,0,0,0,1,0,0,0,0,0]) 
                    elif config == spf8:
                        label=np.array([0,0,0,0,0,0,0,0,1,0,0,0,0]) 
                    elif config == spf9:
                        label=np.array([0,0,0,0,0,0,0,0,0,1,0,0,0]) 
                    elif config == spf10:
                        label=np.array([0,0,0,0,0,0,0,0,0,0,1,0,0]) 
                    elif config == spf11:
                        label=np.array([0,0,0,0,0,0,0,0,0,0,0,1,0]) 
                    elif config == spf12:
                        label=np.array([0,0,0,0,0,0,0,0,0,0,0,0,1]) 
                        
                    labels.append(label)
                    
    return np.asarray(labels)

#------------------------------------------------------------------------------------------------------------------------------    
def reshape_minibatch(minibatch_data): #, minibatch_labels):
    # inputs is a list of numpy 2d arrays.
    # Ouput: 4d tensor of minibatch data and 2d label arrays
    
    l = len(minibatch_data)
    t, f = minibatch_data[0].shape
    #print('Time and frequency', t, f)
    
    reshaped_data = np.empty((l,t,f))
    for i in range(l):
        reshaped_data[i] = minibatch_data[i]
    
    #print('New 3d shape = ', reshaped_data.shape)
    
    # Now convert 3d array to 4d array
    reshaped_data = np.expand_dims(reshaped_data, axis=3)
    #print('New 4d shape = ', reshaped_data.shape)    
    
    # Re-arrange binary labels in one-hot 2 dimensional vector form
    #new_labels = [[1,0] if label == 1 else [0,1] for label in minibatch_labels]
    
    return np.asarray(reshaped_data) #, np.asarray(new_labels)

#------------------------------------------------------------------------------------------------------------------------------    
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    '''
    Generator function for iterating over the dataset.
    
    inputs = list of numpy arrays
    targets = list of numpy arrays that hold labels : TODO , need to fix the labels properly    
    '''
        
    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))    
    np.random.shuffle(indices)    

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):  #total batches                 
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:            
            excerpt = slice(start_idx, start_idx + batchsize)
 
        yield np.asarray(inputs)[excerpt], np.asarray(targets)[excerpt]
    
#------------------------------------------------------------------------------------------------------------------------------

def spectrograms(input_type, data_list,savePath,fft_size,win_size,hop_size,duration):
    
    from audio import compute_spectrogram              
    spectrograms = list()
    print('Computing the spectrograms ..')
    
    with open(data_list, 'r') as f:
        spectrograms = [compute_spectrogram(input_type,file.strip(),fft_size, win_size, hop_size, duration) for file in f]         
    #Save the data as .npz file in savePath
    makeDirectory(savePath)
    outfile = savePath+'/spec'
    with open(outfile,'w') as f:
        np.savez(outfile, spectrograms=spectrograms)
        print('Finished computing spectrogram and saved inside: ', savePath)
                
            
#------------------------------------------------------------------------------------------------------------------------------            
def load_data(path):
    
    specFile = path+'spec.npz'
    #labelFile= path+'labels.npz'
        
    with np.load(specFile) as f:
        data = f['spectrograms']
    #with np.load(labelFile) as f:
    #    labels = f['labels']

    return data #, labels   # we cant tie data and labels together

def compute_global_norm(data,mean_std_file):
    print('Computing Global Mean Variance Normalization from the Data and save on disk..')
    mean,std = audio.compute_mean_std(data)      
    
    with open(mean_std_file, 'w') as f:
        np.savez(mean_std_file, mean=mean, std=std)
        
def normalise_data(data,mean_std_file,normType):
    
    #print('Input to normalise_data function is: ', len(data))
    mean = None
    if normType == 'utterance':        
        print('Utterance based Mean Variance Normalization..')
        newData = list()
        for spect in data:
            mean,std = audio.compute_mean_std(spect)
            inv_std = np.reciprocal(std)
            newData.append((np.asarray(spect)-np.asarray(mean)) * np.asarray(inv_std))
        data=newData
        
    elif normType == 'global_mv':

        print('Performing global mean and variance normalization')
        if(os.path.isfile(mean_std_file)):
            with np.load(mean_std_file) as f:
                mean = f['mean']
                std = f['std']
        else:
            print('************* Mean and std file not found in given path *************** ')
            assert(mean != None)
            

        #instead of dividing by std, its efficient to multiply     
        inv_std = np.reciprocal(std)                              
        data = [(spect-mean) * inv_std for spect in data]        
        
    elif normType == 'global_m':
        
        print('Performing global mean normalization')
        if(os.path.isfile(mean_std_file)):
            with np.load(mean_std_file) as f:
                mean = f['mean']
        else:
            print('************* Mean file not found in given path *************** ')
            assert(mean != None)
    else:
        print('No normalization chosen !')
                    
    return data

#------------------------------------------------------------------------------------------------------------------------------
def prepare_data(basePath,dataType,outPath,inputType='mag_spec',duration=3,targets=2,
                 fs=16000,fft_size=512,win_size=512,hop_size=160):    
    
    print('The spectrogram savepath is: ', outPath)
    
    trainP=basePath+'/ASVspoof2017_train_dev/protocol/ASVspoof2017_train.trn'
    devP=basePath+'/ASVspoof2017_train_dev/protocol/ASVspoof2017_dev.trl'
    #evalP=basePath+'/ASVspoof2017_eval/ASVspoof2017_eval.trl'
    evalP=basePath+'/labels/eval_genFirstSpoof_twoColumn.lab'

    train_list = basePath+'/filelists/train.scp'
    validation_list = basePath+'/filelists/dev.scp'
    evaluation_list = basePath+'/filelists/eval_genFirstSpoof.scp'
                
    train_key  = basePath+'/labels/train.lab'
    validation_key = basePath+'/labels/dev.lab'
    evaluation_key = basePath+'/labels/eval_genFirstSpoof.lab'
    
    splitPath=basePath+'/filelists/eval_split_genuineFirst_Spoof/'    
    labPath=basePath+'/filelists/eval_label_split_genuineFirst_Spoof/label_'            
    splitParts=7
    
    data=list()
    labels=list()
    labelPath=None
    audio_list=None
    savePath = None
    
    if dataType == 'train':
        savePath=outPath+'train/'
        labelPath=trainP
        audio_list=train_list
    elif dataType == 'validation' or dataType=='dev':
        savePath=outPath+'dev/'
        labelPath=devP
        audio_list=validation_list        
    elif dataType == 'test' or dataType == 'eval' or dataType=='evaluation':
        savePath=outPath+'eval/'    
        labelPath=evalP
        audio_list=evaluation_list
        
    assert(audio_list != None)
    assert(labelPath != None)
    assert(savePath != None)
        
    if inputType == 'log_fbank':
        #data = 
        some_function(input_type, audio_list,savePath,fft_size,win_size,hop_size,duration) # todo
    else:
        #data = 
        spectrograms(inputType,audio_list,savePath,fft_size,win_size,hop_size,duration)
            
    #print('After computing spectrogram, length is: ', len(data))
    
    #labels=get_labels_according_to_targets(labelPath, targets)
    
    # Save the labels
    #outfile = savePath+'/labels'
    #with open(outfile,'w') as f:
    #    np.savez(outfile, labels=labels)
    #    print('Saved the labels in: ', savePath)
        
def get_Data_and_labels(dataType, outPath, mean_std_file,specType='mag_spec',duration=3,targets=2,computeNorm=False,
                        normType='global',normalise=True,fs=16000,fft_size=512,win_size=512,hop_size=160):
    
    #TODO : I might want to experiment with various input types, such as mel-spectrogram, chromogram etc
    #specType='magnitude_spectrogram'  #default
    #mel_spectrogram
    #chromogram based on CQT transform
    
    print('The spectrogram savepath is: ', outPath)
    
    trainP='/import/c4dm-datasets/SpeakerRecognitionDatasets/ASVSpoof2017/ASVspoof2017_train_dev/protocol/ASVspoof2017_train.trn'
    devP='/import/c4dm-datasets/SpeakerRecognitionDatasets/ASVSpoof2017/ASVspoof2017_train_dev/protocol/ASVspoof2017_dev.trl'
    #evalP='/import/c4dm-datasets/SpeakerRecognitionDatasets/ASVSpoof2017/ASVspoof2017_eval/ASVspoof2017_eval.trl'
    evalP='/import/c4dm-datasets/SpeakerRecognitionDatasets/ASVSpoof2017/labels/eval_genFirstSpoof_twoColumn.lab'

    train_list = '/import/c4dm-datasets/SpeakerRecognitionDatasets/ASVSpoof2017/filelists/train.scp'
    validation_list = '/import/c4dm-datasets/SpeakerRecognitionDatasets/ASVSpoof2017/filelists/dev.scp'
    evaluation_list = '/import/c4dm-datasets/SpeakerRecognitionDatasets/ASVSpoof2017/filelists/eval_genFirstSpoof.scp'
                
    train_key  = '/import/c4dm-datasets/SpeakerRecognitionDatasets/ASVSpoof2017/labels/train.lab'  
    validation_key = '/import/c4dm-datasets/SpeakerRecognitionDatasets/ASVSpoof2017/labels/dev.lab'    
    evaluation_key = '/import/c4dm-datasets/SpeakerRecognitionDatasets/ASVSpoof2017/labels/eval_genFirstSpoof.lab'   
    
    splitPath='/import/c4dm-datasets/SpeakerRecognitionDatasets/ASVSpoof2017/filelists/eval_split_genuineFirst_Spoof/'    
    labPath='/import/c4dm-datasets/SpeakerRecognitionDatasets/ASVSpoof2017/filelists/eval_label_split_genuineFirst_Spoof/label_'
            
    splitParts=7
    
    data=list()
    labels=list()
    
    if dataType == 'train':
        print('Now in trainining')
        
        mode='training'
        savePath=outPath+'train/'       
        data = spectrograms(train_list,savePath,fft_size,win_size,hop_size,duration)
        print('After computing spectrogram in training set: ', len(data))
        labels=get_labels_according_to_targets(trainP, targets)

        if normalise:
            data=normalise_data(data,mean_std_file,normType,computeNorm)   #mode)
            
    elif dataType == 'validation':
        mode='testing'
        savePath=outPath+'dev/'        
        data = spectrograms(validation_list,savePath,fft_size,win_size,hop_size,duration)
        labels=get_labels_according_to_targets(devP, targets)        
        if normalise:
            data=normalise_data(data,mean_std_file,normType,computeNorm)  #mode)
    
    elif dataType == 'test':        
        mode='testing'
        savePath=outPath+'eval/'
        specPath = savePath

        data = spectrograms(evaluation_list,savePath,fft_size,win_size,hop_size,duration)
        labels=get_labels_according_to_targets(evalP, targets=2)  # For now just 2 targets on eval !
        
        if normalise:
            data=normalise_data(data,mean_std_file,normType,computeNorm)  #mode)

        
    return data, labels                   

def normalise_dataOLD(data,mean_std_file,normType,computeNorm):  #mode):
    
    print('Input to normalise_data function is: ', len(data))
    
    if normType == 'utterance':
        # irrespective of training or test mode, it will perform per utterance normalization
        print('Utterance based Mean Variance Normalization..')
        newData = list()
        for spect in data:
            mean,std = audio.compute_mean_std(spect)
            inv_std = np.reciprocal(std)
            newData.append((np.asarray(spect)-np.asarray(mean)) * np.asarray(inv_std))
        data=newData
        
    elif normType == 'global':
        mean=0
        std=0
        
        #Compute mean and variance along time axis from training set spectrograms
        if computeNorm:   # mode=='training':
            print('Computing Global Mean Variance Normalization from the Data and save on disk..')
            mean,std = audio.compute_mean_std(data)
        
            #Save the mean and variance in disk for later use            
            with open(mean_std_file, 'w') as f:
                np.savez(mean_std_file, mean=mean, std=std)
            
        #Load mean and std from the path provided        
        else:
            print('Loading the precomputed means and std')
            if(os.path.isfile(mean_std_file)):
                with np.load(mean_std_file) as f:
                    mean = f['mean']
                    std = f['std']
            else:
                print('Mean and std file does not exist..computing it now')            

        #instead of dividing by std, its efficient to multiply     
        inv_std = np.reciprocal(std)                              
        data = [(spect-mean) * inv_std for spect in data]
        #Above line throws error with python2.7 for broadcasting
        
    else:
        print('No normalization is performed..')
        
    return data  
    
def load_spectrograms(spec_file):
    with np.load(spec_file) as f:
        spec_data = f['spectrograms']
        #spec_labels = f['labels']
        return spec_data  #, spec_labels 
    
    
def spectrogramsOLD(data_list,savePath,fft_size,win_size,hop_size,duration):
        
    spectrograms = list()
    
    #check if spectrogram exist in savePath. If yes just load it and return
    if os.path.isfile(savePath+'/spec.npz'):
        print('Loading pre-computed spectrograms ...')        
        spectrograms = load_spectrograms(savePath+'/spec.npz')
        
    else: # Compute it and save it and also return to calling function          
            print('Computing the spectrograms ..')
            with open(data_list, 'r') as f:                
                spectrograms = [audio.compute_spectrogram(
                    file.strip(),fft_size, win_size, hop_size, duration) for file in f]
                
            #Save the data as .npz file in savePath
            makeDirectory(savePath)
            outfile = savePath+'/spec'
            with open(outfile,'w') as f:
                np.savez(outfile, spectrograms=spectrograms)
                print('Finished computing spectrogram and saved inside: ', savePath)
                
    return spectrograms    