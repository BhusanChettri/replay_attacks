#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
 Using Russian ARCHITECTURE !!
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
    activation = 'mfm'  #'elu'
    init_type='xavier'

    batch_size = 32
    epochs = 2000 #0
    
    # Regularizer parameters
    use_lr_decay=False        #set this flag for LR decay
    wDecayFlag = False        #whether to perform L2 weight decay or not
    lossPenalty = 0.001       # Using lambda=0.001 .
    applyBatchNorm = False    
    deviceId = "/gpu:0"  
      
    # Adam parameters
    optimizer_type = 'adam'
    b1=0.9
    b2=0.999
    epsilon=0.1
    momentum=0.95
    
    dropout3=1.0  #In russian arch there are only FC1 and output layer, so this not needed. 
    
    #dropout1=[0.3,0.2,0.1]          # we ran this originally ! we will look into this later
    #dropout2=[0.7,0.5,0.4,0.3,0.2]  # for dropout1=0.3, we ran all combination  
    
    dropout1=[0.2,0.1]
    dropout2=[0.4,0.2]
    
    #lambdas = [0.0005, 0.001]
    
    #dropout1=[1.0]
    #dropout2=[1.0]
    #dropout3=1.0
    
    architectures = [5]   # Russian Architecture is 5
    trainingSize = [4]   #in seconds
    lr= 0.0001
               
    targets=2
    fftSize = 2048
    specType='mag_spec'
    padding=True
    
    # Used following paths since I moved the scripts in git and used link so that code are synchronised
    spectrogramPath='/homes/bc305/myphd/stage2/deeplearning.experiment1/spectrograms/'
    tensorboardPath = '/homes/bc305/myphd/stage2/deeplearning.experiment1/CNN3/tensorflow_log_dir/'
    modelPath = '/homes/bc305/myphd/stage2/deeplearning.experiment1/CNN3/models/'
                
    for duration in trainingSize:
        print('Now loading the data !!')
        outPath = spectrogramPath+specType + '/' +str(fftSize)+ 'FFT/' + str(duration)+ 'sec/'
        mean_std_file = outPath+'train/mean_std.npz'
                
        # Load training data, labels and perform norm
        tD = dataset.load_data(outPath+'train/')
        tL=dataset.get_labels_according_to_targets(trainP, targets)

        if not os.path.exists(mean_std_file):
            print('Computing Mean_std file ..')
            dataset.compute_global_norm(tD,mean_std_file)                        
        
        print('Shape of labels: ',tL.shape)
        
        #tD = dataset.normalise_data(tD,mean_std_file,'utterance')    # utterance level        
        tD = dataset.normalise_data(tD,mean_std_file,'global_mv') # global
        
         # Load dev data, labels and perform norm
        devD = dataset.load_data(outPath+'dev/')        
        devL = dataset.get_labels_according_to_targets(devP, targets)        
        #devD = dataset.normalise_data(devD,mean_std_file,'utterance')
        devD = dataset.normalise_data(devD,mean_std_file,'global_mv')
                        
        ### We are training on TRAIN set and validating on DEV set
        t_data = tD
        t_labels = tL
        v_data = devD
        v_labels = devL              
        
        for dropout in dropout1:
            architecture=architectures[0]
            
            for drop in dropout2:
                
                hyp_str ='arch'+str(architecture)+'_keep_'+ str(dropout)+'_'+str(drop)+'_'+str(duration)+'sec'
                
                log_dir = tensorboardPath+'/rusCNN_max2000epochs/'+ hyp_str
                model_save_path = modelPath + '/rusCNN_max2000epochs/'+ hyp_str
                logfile = model_save_path+'/training.log'
                
                figDirectory = model_save_path     
                makeDirectory(model_save_path)
                print('Training model with ' + str(duration) + ' sec data and cnnModel' + str(architecture))
                
                tLoss,vLoss,tAcc,vAcc=model.train(architecture,fftSize,padding,duration,t_data, t_labels,v_data,v_labels,activation,lr,use_lr_decay,epsilon,b1,b2,momentum,optimizer_type,dropout,drop,dropout3,model_save_path,log_dir,logfile,wDecayFlag,lossPenalty,applyBatchNorm,init_type,epochs,batch_size,targets)
                
                #plot_2dGraph('#Epochs', 'Avg CE Loss', tLoss,vLoss,'train_ce','val_ce', figDirectory+'/loss.png')
                #plot_2dGraph('#Epochs', 'Avg accuracy', tAcc,vAcc,'train_acc','val_acc',figDirectory+'/acc.png')
                plot_2dGraph('#Epochs', 'Val loss and accuracy', vLoss,vAcc,'val_loss','val_acc',figDirectory+'/v_ls_acc.png')

                
trainCNN_on_trainData()


