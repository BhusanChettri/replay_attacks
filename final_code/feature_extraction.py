# Extract features from the CNN model which will be subsequently used to train another classifier
# such as GMM or SVM

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Load standard modules
from __future__ import print_function
import sys
import os
import io
import numpy as np
import tensorflow as tf
import shutil

from optparse import OptionParser
from network import init_weights
from network import bias_variable
from utility import makeDirectory
#from model import load_model

# Load userdefined modules
import audio
import dataset
import model
import nn_architecture
import nn_architecture_rus as nn_r
import nn_architecture_birds as nn_b
import numpy as np

#-------------------------------------------------------------------------------------------------------------
def getSplittedDataSet(data, labels, batch_size):
    # This is just created to enable extracting features in batch without missing any data items
    # However we need to ensure that we do not shuffle the data items and the labels
    # Otherwise this will not work as we are not keeping track of labels after it is being read in 
    # the begining.
    
    dataList=list()
    labelList=list()    
    dataCount = getLeftBatchSize(data, batch_size)
    start=0
    end=0
    
    for count in dataCount:
        if count != 0:
            dataBatch = data[start:count+start]
            labelBatch = labels[start:count+start]
            dataList.append(dataBatch)
            labelList.append(labelBatch)
            start += count                
            #print(count)
            
    return dataList, labelList

#-------------------------------------------------------------------------------------------------------------
def makeFileNameList(file):
    # This function will create list of filename from the supplied dataset file    
    with open(file, 'r') as f:
        return [os.path.basename(line.strip()).split('.')[0] for line in f]       
#-------------------------------------------------------------------------------------------------------------
def loadFeatures(filename):            
    print('Loading features')
    
    if(os.path.isfile(filename)):
        with np.load(filename) as f:
            features = f['features']            
        return features
    
    else:
        print('No parameters found')                 
        
#-------------------------------------------------------------------------------------------------------------        
def saveFeatures(features, outfile):
    with open(outfile,'w') as f:
        np.savez(outfile, features=features)
#-------------------------------------------------------------------------------------------------------------        

def getLeftBatchSize(datalist, batch):
    
    totalLength=len(datalist)
    t=int(totalLength/batch)
    dt = totalLength - t*batch    
    totalDataThatWillbeUsed = t*batch
    
    print('Total data items used with this batch size = ', totalDataThatWillbeUsed)
    print('Total leftover data will be used as a single batch of: ', dt)
    
    return totalDataThatWillbeUsed, dt
    
#-------------------------------------------------------------------------------------------------------------    
def computeModelScore(model_prediction, apply_softmax=True):
    # model_prediction is the output from the output layer which is in logits.
    # we apply softmax to get probability depending upon flag variable being passed
    
    if apply_softmax:
        prediction = tf.nn.softmax(model_prediction)
    else:
        prediction = model_prediction
        
    return prediction   #writeOutput(prediction)


#-------------------------------------------------------------------------------------------------------------    

def extract_CNN_Features(featType,dataType,architecture,trainSize,test_data,test_labels, model_path, n_model, activation, 
                         init_type,targets,fftSize,padding,batch_size=1):
    
    # Note: The network that was trained with 4 targets for instance cannot be used with 2 targets to extract features
    # as I was thinking. Things messes up. Therefore we could only use it for feature extraction. In case of eval data
    # as we do not have labels, just create dummy labels of 4 dimension for feature extraction.       

    if dataType=='test':   ##This is just a trick for extracting features from CNN
        targets=2
    
    #print('Datatype is = ', dataType)
    #print('Targets = ', targets)
        
    print('Reset TF graph and load trained models and session..')        
    tf.reset_default_graph()
    
    t = trainSize*100  #Default is this
                                  
    if input_type=='mel_spec':
        f= 80
    elif input_type=='cqt_spec':
        f = 84        
        if augment:
            t=47
        else:
            t=47     ## This needs to be fixed. At the moment for CQT, we have 47 as time dimension
        
    elif input_type=='mag_spec':
        
        if fftSize == 512:
            f = 257
        elif fftSize == 256:
            f = 129
        elif fftSize == 1024:
            f = 513
        elif fftSize == 2048: 
            f = 1025
                
    input_data = tf.placeholder(tf.float32, [None,t, f,1])  #make it 4d tensor
    true_labels = tf.placeholder(tf.float32, [None,targets], name = 'y_input')
            
    # Placeholders for droput probability
    keep_prob1 = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)
    keep_prob3 = tf.placeholder(tf.float32)
    
    if activation == 'relu':
        act = tf.nn.relu
    elif activation == 'elu':
        act = tf.nn.elu
    elif activation == 'crelu':
        act = tf.nn.crelu
    elif activation == 'mfm':
        act = 'mfm'

    num_classes=targets
    
    if architecture == 1:
        extractor, model_prediction,network_weights,activations,biases= nn_architecture.cnnModel1(dataType,trainSize,input_data, act,init_type,num_classes,fftSize,padding,keep_prob1,keep_prob2,keep_prob3)
        
    elif architecture == 2:
        extractor, model_prediction,network_weights,activations,biases= nn_b.cnnModel1(trainSize,input_data, act,init_type,num_classes,fftSize,padding,keep_prob1,keep_prob2,keep_prob3)
        
    elif architecture == 3:
        extractor, model_prediction,network_weights,activations,biases= nn_b.cnnModel2(trainSize,input_data, act,init_type,num_classes,fftSize,padding,keep_prob1,keep_prob2,keep_prob3)
        
    elif architecture == 5:
        extractor, model_prediction,network_weights,activations,biases= nn_r.cnnModel5(trainSize,input_data, act,init_type,num_classes,fftSize,padding,keep_prob1,keep_prob2,keep_prob3)


    modelScore = computeModelScore(model_prediction, apply_softmax=True)    
    
    total_batches = int(len(test_data)/batch_size)   #+1
    print('Total_batches on dataset = ', total_batches) 
            
    #Load trained session and model parameters    
    sess, saver = model.load_model(model_path, n_model)
    print('Model parameters loaded succesfully !!')
    
    featureList = list()
    scoreList = list()
    
    # Now prepare training data generator to iterate over mini-batches and run training loop
    batch_generator = dataset.iterate_minibatches(test_data, test_labels, batch_size, shuffle=False)
    
    print('Extracting features for total audio files = ', len(test_data))
    for i in range(total_batches):                
        data, labels = next(batch_generator)        
        data = dataset.reshape_minibatch(data) #, labels)
                
        if featType == 'bottleneck':            
            features = sess.run([extractor] , feed_dict={input_data:data, true_labels:labels,keep_prob1: 1.0,
                                                                keep_prob2: 1.0, keep_prob3: 1.0})
            featureList.append(features) # use append             
        else:
            scores = sess.run([modelScore] , feed_dict={input_data:data, true_labels:labels,keep_prob1: 1.0,
                                                                keep_prob2: 1.0, keep_prob3: 1.0})            
            print('Printing 5 scores in this batch:')
            print(scores[0:5])
            scoreList.append(scores) # use append only
            
    if featType == 'bottleneck':
        return featureList
    else:
        return scoreList
    
#-------------------------------------------------------------------------------------------------------------

def getFeatures(featType,dataType,data,labels,batch_size,model_path,n_model,activation,init_type,targets,fftSize,padding,
                architecture,trainSize):

    featureList=[]    
    dataList, labelList = getSplittedDataSet(data, labels, batch_size)    
    
    for i in range(len(dataList)):
        
        print('Datalist length = ',len(dataList[i]))
        print('Labels shape: ', labelList[0][0].shape)
        
        if len(dataList[i]) < batch_size:
            batch=len(dataList[i])
        else:
            batch = batch_size
            
        feats = extract_CNN_Features(featType,dataType,architecture,trainSize,dataList[i],labelList[i],
                                     model_path,n_model,activation,init_type,targets,fftSize,padding,batch)
                        
        featureList.extend(feats)  #use extend to keep in same list
                                            
    return featureList  #this could be scores or bottleneck features depending upon featType parameter
