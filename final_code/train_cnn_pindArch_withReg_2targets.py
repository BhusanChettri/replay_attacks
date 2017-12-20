#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
 Main function that calls module to train a Convolutional Neural Network
 for detecting between replayed and genuine speech.
'''

# Load standard modules
from __future__ import print_function
import sys
import os
import io
import numpy as np

from optparse import OptionParser
from plotGraph import plot_entropy_loss
from plotGraph import plot_2dGraph
from utility import makeDirectory
from dataset import load_data
from dataset import compute_global_norm

# Load userdefined modules
import audio
import dataset
import model
  
def trainCNN_on_trainData():

    #CNN Training parameters
    activation = 'mfm'  #choose activation: mfm,elu, relu, mfsoftmax, tanh ?
    init_type='xavier'  #'truncated_normal' #'xavier'  #or 'truncated_normal'

    batch_size = 32
    epochs = 1000
    
    # Regularizer parameters
    use_lr_decay=False        #set this flag for LR decay
    wDecayFlag = False         #whether to perform L2 weight decay or not
    lossPenalty = 0.001       # Using lambda=0.001 .
    applyBatchNorm = False    
    deviceId = "/gpu:0"  
      
    # Adam parameters
    optimizer_type = 'adam'
    b1=0.9
    b2=0.999
    epsilon=0.1
    momentum=0.95
    dropout1=0.1                  #for input to first FC layer
    dropout2=0.2                  #for intermediate layer input    
    drops=[0.3]                   # 50% dropout the inputs of FC layers
    lambdas = [0.0005, 0.001]
    
    architectures = [1]
    trainingSize = [1]   #in seconds
    lr= 0.0001
           
    targets=2
           
    #specType='mag_spec'     #lets try loading mag_spec    
    #inputTypes=['mel_spec'] #'mag_spec'  #'cqt_spec'   ## Running on Hepworth !    
    #inputTypes=['mag_spec']    # Not Run yet
    #inputTypes=['cqt_spec']    # Not Run yet
    
    inputTypes=['mag_spec','cqt_spec','mel_spec']
    padding=True
    
    augment = True 
    trainPercentage=1.0    #Each training epoch will see only 50% of the original data at random !
    valPercentage=0.8   
            
    if augment:
        spectrogramPath='/homes/bc305/myphd/stage2/deeplearning.experiment1/spectrograms_augmented/1sec_shift/'
    else:
        spectrogramPath='/homes/bc305/myphd/stage2/deeplearning.experiment1/spectrograms/'        
            
    # Used following paths since I moved the scripts in git and used link so that code are synchronised
    tensorboardPath = '/homes/bc305/myphd/stage2/deeplearning.experiment1/CNN3/tensorflow_log_dir/'
    modelPath = '/homes/bc305/myphd/stage2/deeplearning.experiment1/CNN3/models_augmented/'
                
    #for duration in trainingSize:
    duration=1
    fftSize=512
    
    for specType in inputTypes:
               
        #if specType == 'mel_spec' or specType == 'cqt_spec':            
        #    fftSize=512
        #else:            
        #    fftSize=256
                
        #print('Now loading the data with FFT size: ', fftSize)
        outPath = spectrogramPath +specType + '/' +str(fftSize)+ 'FFT/' + str(duration)+ 'sec/'
        mean_std_file = outPath+'train/mean_std.npz'
                
        # Load training data, labels and perform norm
        tD,tL = dataset.load_data(outPath+'train/')
        tL = dataset.get_labels_according_to_targets(tL, targets)
        
        assert(len(tD)==len(tL))
        
        if not os.path.exists(mean_std_file):
            print('Computing Mean_std file ..')
            dataset.compute_global_norm(tD,mean_std_file)
            
        # We will try utterance based norm later   
        #tD = dataset.normalise_data(tD,mean_std_file,'utterance')    # utterance level      
        tD = dataset.normalise_data(tD,mean_std_file,'global_mv')    # global
                        
        # Now take only 80% of the new augmented data to use for validation
        # Just to save some time during training
        devD,devL = dataset.get_random_data(outPath+'dev/',batch_size,valPercentage)                
        devL = dataset.get_labels_according_to_targets(devL, targets)        
        assert(len(devD)==len(devL))                                
                        
        #devD = dataset.normalise_data(devD,mean_std_file,'utterance')
        devD = dataset.normalise_data(devD,mean_std_file,'global_mv')                                
                
        ### We are training on TRAIN set and validating on DEV set        
        t_data = tD
        t_labels = tL
        v_data = devD
        v_labels = devL                
        
        print('Training model on ', specType)

        for dropout in drops:                  # dropout 1.0 and 0.5 to all inputs of DNN
            architecture = architectures[0]
            #penalty=0.001  #this is not used thought at the moment
            
            for penalty in lambdas:
            #for targets in target_list:
                                                
                hyp_str='keep_'+str(dropout1)+'_'+str(dropout2)+'_'+str(dropout)+'_'+str(specType)#+'_targets'+str(targets)
                
                log_dir = tensorboardPath+ '/model1_max1000epochs/'+ hyp_str
                model_save_path = modelPath + '/model1_max1000epochs/'+ hyp_str
                logfile = model_save_path+'/training.log'
                
                figDirectory = model_save_path        
                makeDirectory(model_save_path)
                                                
                tLoss,vLoss,tAcc,vAcc=model.train(specType,architecture,fftSize,padding,duration,t_data,t_labels,
                                                  v_data,v_labels,activation,lr,use_lr_decay,epsilon,b1,b2,momentum,
                                                  optimizer_type,dropout1,dropout2,dropout,model_save_path,log_dir,
                                                  logfile,wDecayFlag,penalty,applyBatchNorm,init_type,epochs,batch_size,
                                                  targets,augment)#,trainPercentage,valPercentage)                                                                                                                                        
                #plot_2dGraph('#Epochs', 'Avg CE Loss', tLoss,vLoss,'train_ce','val_ce', figDirectory+'/loss.png')
                #plot_2dGraph('#Epochs', 'Avg accuracy', tAcc,vAcc,'train_acc','val_acc',figDirectory+'/acc.png')
                #plot_2dGraph('#Epochs', 'Val loss and accuracy', vLoss,vAcc,'val_loss','val_acc',figDirectory+'/v_ls_acc.png')
                        
trainCNN_on_trainData()

'''
TODO:
Before running the full model, first test with 1 epoch to ensure that the inside-code feature extraction and scoring pipeline is working well. Once it is okay, run these models in full epochs of 2000 !!
'''

# TODAY
# 1. Reextract mel-spectrogram