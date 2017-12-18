from __future__ import print_function
import numpy as np
import audio
import os
from utility import makeDirectory

def get_labels_according_to_targets(protocalList, targets=2):
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
    
    if targets==2:
        labels = [[1,0] if line.strip().split(' ')[1] == 'genuine' else [0,1] for line in protocalList]        
    else:
        for line in protocalList:
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
def reshape_minibatch(minibatch_data):
    # inputs is a list of numpy 2d arrays.
    # Ouput: 4d tensor of minibatch data and 2d label arrays
    
    l = len(minibatch_data)
    t, f = minibatch_data[0].shape
    
    #print('In reshape minibatch, Time and frequency', t, f)
    
    reshaped_data = np.empty((l,t,f))
    for i in range(l):        
        reshaped_data[i] = minibatch_data[i]
                    
    # Now convert 3d array to 4d array
    reshaped_data = np.expand_dims(reshaped_data, axis=3)
    #print('Returning from reshape_minibatch')
            
    return np.asarray(reshaped_data)

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

def load_data(file):
    #if augment
    with np.load(file+'spec.npz') as f:
        spec_data = f['spectrograms']
        spec_labels = f['labels']
        return spec_data,spec_labels 

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

def augment_data(data,label,data_window=100,input_type='mel_spec',shift=10):
    '''
    Inputs: 
       data = original data matrix in TxF format. Rows specifies frames and columns frequency.
       data_window = how many frames to keep in one matrix
       input_type = either CQT or MEl_SPEC
       shift = 10 by default. If a frame is 32ms, then 10 shift corresponds to 320ms
       
    Outputs:
       a list of matrices obtained from the original matrix using sliding window mechanism, this is kind
       of a data augmentation technique for producing many matrices.       
       '''
    
    dataList = list()
    labelList = list()        
    
    t,f = data.shape            
    window = data_window  
    
    #label = label_line.split(' ')[1]
    
    assert(t > window)
            
    start=0    
    for i in range(window, t, shift):
                
        new_data = data[start:i]
        start += shift
        dataList.append(new_data)
        labelList.append(label)
                
    return dataList,labelList


def spectrograms(input_type,data_list,labelFile,savePath,fft_size,win_size,hop_size,duration,data_window=100,
                 window_shift=10,augment=True,save=True,minimum_length=1):
                    
    from audio import compute_spectrogram              

    spectrograms = list()
    labels = list()
        
    print('Computing the ' + input_type + ' spectrograms !!')    
    with open(data_list, 'r') as f:
        spectrograms = [compute_spectrogram(input_type,file.strip(),fft_size,win_size,hop_size,duration,augment,minimum_length)
                        for file in f]                 
    # Get the labels into a list and save it along with the spectrograms
    with open(labelFile,'r') as f:
        #labels = [1 if line.strip().split(' ')[1] == 'genuine' else 0 for line in f]
        labels = [line.strip() for line in f]     
                        
    if augment:
        new_data = list()
        new_labels = list()
        
        assert(len(labels) == len(spectrograms))
        print('Now performing augmentation using sliding window mechanism on original spectrogram .... ')
        
        for i in range(len(spectrograms)):    
            d,l = augment_data(spectrograms[i],labels[i],data_window,input_type,window_shift)            
            new_data.extend(d) # extend the list rather than adding it into a new list
            new_labels.extend(l)
            
        spectrograms = new_data
        labels = new_labels
    
    if save:  
        from helper import makeDirectory
        makeDirectory(savePath)
        outfile = savePath+'/spec'
        with open(outfile,'w') as f:
            np.savez(outfile, spectrograms=spectrograms, labels=labels)
        print('Finished computing spectrogram and saved inside: ', savePath)
    
    # We always save the spectrograms, coz we run different experiments on same data.
    # While loading spectrogram check if its augmented one or simple one    
    
def prepare_data(basePath,dataType,outPath,inputType='mag_spec',duration=3,
                 fs=16000,fft_size=512,win_size=512,hop_size=160,data_window=100,window_shift=10,
                 augment=True,save=True,minimum_length=1): 
        
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
        some_function(input_type, audio_list,labelPath,savePath,fft_size,win_size,hop_size,duration,
                      data_window,window_shift,augment,save,minimum_length)
    else:        
        spectrograms(inputType,audio_list,labelPath,savePath,fft_size,win_size,hop_size,duration,
                     data_window,window_shift,augment,save,minimum_length)