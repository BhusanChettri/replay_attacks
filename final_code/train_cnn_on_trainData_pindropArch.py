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
import tensorflow as tf

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

basePath='/import/c4dm-datasets/SpeakerRecognitionDatasets/ASVSpoof2017/'
trainP=basePath+'/ASVspoof2017_train_dev/protocol/ASVspoof2017_train.trn'
devP=basePath+'/ASVspoof2017_train_dev/protocol/ASVspoof2017_dev.trl'
evalP=basePath+'/labels/eval_genFirstSpoof_twoColumn.lab'
   
def trainCNN_on_trainData():

    #CNN Training parameters
    activation = 'mfm'  #choose activation: mfm,elu, relu, mfsoftmax, tanh ?
    init_type='xavier'  #'truncated_normal' #'xavier'  #or 'truncated_normal'

    batch_size = 32
    epochs = 2000
    
    # Regularizer parameters
    use_lr_decay=False        #set this flag for LR decay
    wDecayFlag = True         #whether to perform L2 weight decay or not
    lossPenalty = 0.001       # Using lambda=0.001 .
    applyBatchNorm = False    
    deviceId = "/gpu:0"  
      
    # Adam parameters
    optimizer_type = 'adam'
    b1=0.9
    b2=0.999
    epsilon=0.1
    momentum=0.95
    dropout1=0.1                 #for input to first FC layer  
    dropout2=0.2                 #for intermediate layer input    
    drops=[0.4]
    lambdas = [0.0005, 0.001]
    
    architectures = [1]
    trainingSize = [1]   #in seconds
    lr= 0.0001
           
    targets=2
    fftSize = 512
    specType='mag_spec'    
    padding=True
                
    for duration in trainingSize:
        print('Now loading the data !!')
        outPath = '../../spectrograms/'+specType + '/' +str(fftSize)+ 'FFT/' + str(duration)+ 'sec/'
        mean_std_file = outPath+'train/mean_std.npz'
                
        # Load training data, labels and perform norm
        tD = dataset.load_data(outPath+'train/')
        tL=dataset.get_labels_according_to_targets(trainP, targets)
        dataset.compute_global_norm(tD,mean_std_file)
        
        print('Shape of labels: ',tL.shape)
        
        #tD = dataset.normalise_data(tD,mean_std_file,'utterance')    # utterance level        
        tD = dataset.normalise_data(tD,mean_std_file,'global_mv') # global
        #print('Norm td: max and min are ', np.max(tD))
        
         # Load dev data, labels and perform norm
        devD = dataset.load_data(outPath+'dev/')        
        devL = dataset.get_labels_according_to_targets(devP, targets)        
        #devD = dataset.normalise_data(devD,mean_std_file,'utterance')
        #print('first Norm dev: max and min are ', np.max(devD))
        devD = dataset.normalise_data(devD,mean_std_file,'global_mv')
        #print('Norm dev: max and min are ', np.max(devD))
                
        trainSize=str(duration)+'sec'  ##may be change this in model.py also !
        
        ### We are training on TRAIN set and validating on DEV set
        t_data = tD
        t_labels = tL
        v_data = devD
        v_labels = devL                
             
        for dropout in drops:
            architecture = architectures[0]
            
            for penalty in lambdas:
                
                hyp_str ='_cnnModel'+str(architecture)+'_keepProb_0.1_0.2_'+ str(dropout)+'_'+str(penalty)
                
                log_dir = '../tensorflow_log_dir/model1_max2000epochs_L2/'+ hyp_str
                model_save_path = '../models/model1_max2000epochs_L2/'+ hyp_str
                logfile = model_save_path+'/training.log'
                figDirectory = model_save_path        
                makeDirectory(model_save_path)
                print('Training model with ' + str(duration) + ' sec data and cnnModel' + str(architecture))
                
                tLoss,vLoss,tAcc,vAcc=model.train(architecture,fftSize,padding,trainSize,t_data, t_labels,v_data,v_labels,activation,lr,use_lr_decay,epsilon,b1,b2,momentum,optimizer_type,dropout1,dropout2,dropout,model_save_path,log_dir,logfile,wDecayFlag,penalty,applyBatchNorm,init_type,epochs,batch_size,targets)
                                                                                        
                #plot_2dGraph('#Epochs', 'Avg CE Loss', tLoss,vLoss,'train_ce','val_ce', figDirectory+'/loss.png')
                #plot_2dGraph('#Epochs', 'Avg accuracy', tAcc,vAcc,'train_acc','val_acc',figDirectory+'/acc.png')
                plot_2dGraph('#Epochs', 'Val loss and accuracy', vLoss,vAcc,'val_loss','val_acc',figDirectory+'/v_ls_acc.png')
                
trainCNN_on_trainData()



