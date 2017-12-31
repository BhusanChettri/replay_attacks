#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
 Run prediction and obtain model scores and also features from the CNN !!
'''

# Load standard modules
import sys
import os
import io
import numpy as np

import dataset
#from dataset import get_Data_and_labels
from feature_extraction import getFeatures
from feature_extraction import saveFeatures

from utility import makeDirectory

def write_scores_to_file(prediction, after_decimal=5, outfile='prediction.txt'):
    print('Prediction list length = ', len(prediction))
    posteriors = [vector for scoreList in prediction for scores in scoreList for vector in scores]
    with open(outfile, 'w') as f:
        for probs in posteriors:
            gen=probs[0]
            spoof=probs[1]
            
            #if gen == 1.:
            #    gen=0.98
            #    spoof=0.02
            #elif gen == 0.:
            #    gen=0.02
            #    spoof=0.98
 
            #if gen == 0.:    #just to avoid log(0)
            #   gen = 0.01
             
            #elif spoof == 0.:
            #   spoof = 0.01
             
            score = np.log(gen) - np.log(spoof)
            f.write(str(gen)+ ' '+ str(spoof) + ' '+ str(score)+ '\n')
            #f.write("%.4f\n" % (score))
            
def run_prediction(model_path,featType,dataType,protocal,inputPath,mean_std_file,outBase,batch_size=100,activation='elu',
                   init_type='xavier',targets=2,fftSize=256,architecture=2,duration=1,padding=True,n_model=None,
                   inputType='mag_spec',augment=True):
    
    # Extract Features from Training set
    #print('Extracting ' + featType + ' for the ' + dataType + 'set')
    
    print('outBase in run_prediction is: ', outBase)
    
    
    data,lab = dataset.load_data(inputPath+ dataType+'/')
    
    #data = dataset.normalise_data(data,mean_std_file,'utterance')
    data = dataset.normalise_data(data,mean_std_file,'global_mv')    
    labels = dataset.get_labels_according_to_targets(lab, targets)
        
    featureList=getFeatures(featType,inputType,data,labels,batch_size,model_path,n_model,activation,init_type,targets,
                          fftSize,padding,architecture,duration,augment)
            
    if featType == 'bottleneck':
        makeDirectory(outBase+'/features/')#  dataType)       
        saveFeatures(featureList,outBase+'/features/'+dataType)   #saves as train.npz, dev.npz etc
    elif featType == 'scores':
        makeDirectory(outBase+'/predictions/')
        write_scores_to_file(featureList, outfile=outBase+'/predictions/'+ str(dataType)+'_prediction.txt')
    else:
        print('PLEASE CHOSE CORRECT PARAM !!')

#--------------------------------------------------------------------------------------------------------------------

def get_scores_and_features(model_path,batch_size=100,init_type='xavier',activation='elu',normType='global',normalise=True,
                            architecture=2,specType='mag_spec',targets=2,fftSize=256,duration=1,padding=True,
                            featType='scores',augment=False):
    
    basePath='/import/c4dm-datasets/SpeakerRecognitionDatasets/ASVSpoof2017/'
    trainProtocal=basePath+'/ASVspoof2017_train_dev/protocol/ASVspoof2017_train.trn'
    devProtocal=basePath+'/ASVspoof2017_train_dev/protocol/ASVspoof2017_dev.trl'
    evalProtocal=basePath+'/labels/eval_genFirstSpoof_twoColumn.lab'
    
    mode='testing'
    n_model = None
    
    print('The specType passed is: ', specType)
            
    if augment:
        if str(specType).endswith('spec'): 
            spectrogramPath='/homes/bc305/myphd/stage2/deeplearning.experiment1/spectrograms_augmented/1sec_shift/'    
        else:
            spectrogramPath = '/homes/bc305/myphd/stage2/deeplearning.experiment1/features_1sec_shift/'  
      
    else:
        if str(specType).endswith('spec'): 
            spectrogramPath='/homes/bc305/myphd/stage2/deeplearning.experiment1/spectrograms/'    
        else:
            spectrogramPath='/homes/bc305/myphd/stage2/deeplearning.experiment1/features/'     
                                       
    
    if str(specType).endswith('spec'):
        inputPath = spectrogramPath + specType + '/' +str(fftSize)+ 'FFT/' + str(duration)+ 'sec/'
    else:
        inputPath = spectrogramPath + specType + '/'
        
    
    print('Caution: We have used train-set computed mean_std file at the moment !! Be sure !!')
    mean_std_file = inputPath+'train/mean_std.npz'
    
    for feat in featType:        
        outputPath = model_path
        
        print('Performing ' + feat + ' extraction using trained model on train, dev and eval data')

        run_prediction(model_path,feat,'train',trainProtocal,inputPath,mean_std_file,outputPath,batch_size,activation,
                       init_type,targets,fftSize,architecture,duration,padding,n_model,specType,augment)                
        
        #run_prediction(model_path,feat,'dev',trainProtocal,inputPath,mean_std_file,outputPath,batch_size,activation,
        #               init_type,targets,fftSize,architecture,duration,padding,n_model,specType,augment)
        
        print('Now extracting on Eval set !')
    
        #run_prediction(model_path,feat,'eval',trainProtocal,inputPath,mean_std_file,outputPath,batch_size,activation,
        #               init_type,targets,fftSize,architecture,duration,padding,n_model,specType,augment)
        
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------

'''


model_path = '../models/birdsArch_max2000epochs/_cnnModel2_keepProb_0.6_0.50.5lr0.0003/'
init_type='xavier'
activation='elu'
normType = 'global'
normalise=True
architecture = 2
specType='mag_spec'
targets=2
fftSize=256
duration=1
padding=True
batch_size = 100
featType=['scores','bottleneck']
get_scores_and_features(model_path,batch_size,init_type,activation,normType,normalise,architecture,specType,targets,
                        fftSize,duration,padding,featType) 

## NOTE: I can call the above function directly from model.py so that everything gets save in the same model directory after model training !!

'''     # these are called from run_extractor.py
