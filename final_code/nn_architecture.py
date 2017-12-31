#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Network.py

This file holds various functions of a neural network within the tensorflow.
We may change it later on when implementing a class.
'''

# Load standard modules
from __future__ import print_function
import sys
import os
import io
import shutil
import numpy as np
import tensorflow as tf

# Load userdefined modules
from network import maxPool2x2
from network import init_weights
from network import bias_variable
from network import variable_summaries
from network import fc_layer
from network import conv_layer
from network import drop_layer
from network import batchnorm
from network import fc_layer_noAct
from network import conv_layer_noAct

#---------------------------------------------------------------------------------------------------------------
def cnnModel1(input_type,trainSize,input_placeholder,activation,init,targets,fftSize,padding,keep_prob1, keep_prob2, keep_prob3):
    
    # So I will have to come back to this again for code cleaning.
    # Code affected would be: feature_extraction.py, extract_cnn_scores.py, nn_architecture.py, extract_cnn_features.py
    t=trainSize
    
    trainSize=str(trainSize)+'sec'
            
    if input_type=='mel_spec':
        f= 80
    elif input_type=='cqt_spec':
        f = 84        
    elif input_type=='mag_spec':
        
        if fftSize == 512:
            f = 257
        elif fftSize == 256:
            f = 129
        elif fftSize == 1024:
            f = 513
        elif fftSize == 2048: 
            f = 1025
    else:
        concatenate=False
        if concatenate:
            f = 80      # when two types of features are concatenated (eg CQCC+SCMC)
        else:
            f = 40      # just the delta+acceleration (40 dimensional)    
                        
    weight_list = list()
    activation_list = list()
    bias_list = list()
    
    if activation=='mfm':
        fc_input= f*64   #6448 #1*257*64 = 16448
        in_conv2 = 64
        in_conv3 = 64
        in_conv4 = 64
        in_fc2 = 128
        in_fc3 = 128
        in_outputLayer = 128
        
    else:
        fc_input= f*128  #32896 # 1*257*128
        in_conv2 = 128
        in_conv3 = 128
        in_conv4 = 128
        in_fc2 = 256
        in_fc3 = 256
        in_outputLayer = 256
    print('======================== CNN ARCHITECTURE ==============================\n')    
                               
    #Convolution layer1,2,3    
    conv1,w1,b1 = conv_layer(input_placeholder, [3,f,1,128], [128], [1,1,1,1],'conv1',padding,activation,init)
    weight_list.append(w1)
    bias_list.append(b1)
    print('Conv1 ', conv1)
    
    conv2,w2,b2 = conv_layer(conv1, [3,1,in_conv2,128], [128], [1,1,1,1],'conv2', padding,activation,init)
    weight_list.append(w2)
    bias_list.append(b2)    
    print('Conv2 ', conv2)
    
    conv3,w3,b3 = conv_layer(conv2, [3,1,in_conv3,128], [128], [1,1,1,1],'conv3', padding,activation,init)
    weight_list.append(w3)
    bias_list.append(b3)    
    print('Conv2 ', conv3)
    
    if input_type == 'cqt_spec':
        time_dim = 32
    else:
        time_dim = t*100    
    
    #Max-pooling layer over time    
    pool1 = maxPool2x2(conv3, [1,time_dim,1,1], [1,time_dim,1,1])
    print('Pool1 layer shape = ', pool1)
    
    #100x257x64 is input to maxpool
    #output = 1X257x64
    # 1*257*64 = 16448

    # Dropout on the huge input from Conv layer    
    flattened = tf.reshape(pool1, shape=[-1,fc_input])
    dropped_1 = drop_layer(flattened, keep_prob1, 'dropout1')
    
    # Fully connected layer 1 with 256 neurons but gets splitted into 128 due to MFM
    fc1,w4,b4, = fc_layer(dropped_1, fc_input, 256, 'FC_Layer1', activation)
    weight_list.append(w4)
    bias_list.append(b4)
    
    print('Shape of FC1 = ', fc1.shape)
    
    # Dropout followed by FC layer with 256 neurons but gets splitted into 128 due to MFM
    dropped_2 = drop_layer(fc1, keep_prob2, 'dropout2')        
    fc2,w5,b5, = fc_layer(dropped_2, in_fc2, 256, 'FC_Layer2', activation)
    weight_list.append(w5)
    bias_list.append(b5)
    
    print('Shape of FC2 = ', fc2.shape)

    # Dropout followed by FC layer with 256 neurons but gets splitted into 128 due to MFM
    dropped_3 = drop_layer(fc2, keep_prob2, 'dropout3')        
    fc3,w6,b6, = fc_layer(dropped_3, in_fc3, 256, 'FC_Layer3', activation)
    weight_list.append(w6)
    bias_list.append(b6)
    
    print('Shape of FC3 = ', fc3.shape)

    #Output layer: 2 neurons. One for genuine and one for spoof. Dropout applied first
    dropped_4 = drop_layer(fc3, keep_prob3, 'dropout4')
    output,w7,b7 = fc_layer(dropped_4, in_outputLayer, targets, 'Output_Layer', 'no-activation')  #get raw logits        
        
    weight_list.append(w7)
    bias_list.append(b7)            
                
    print('Output layer shape = ', output.shape)
    print('======================== CNN ARCHITECTURE ==============================\n')
    
    
    return fc3, output, weight_list, activation_list, bias_list
#---------------------------------------------------------------------------------------------------------------


def cnnModel2(input_type,trainSize,input_placeholder,activation,init,targets,fftSize,padding,keep_prob1, keep_prob2, keep_prob3):
    
    t=trainSize
    
    trainSize=str(trainSize)+'sec'
            
    if input_type=='mel_spec':
        f= 80
    elif input_type=='cqt_spec':
        f = 84        
    elif input_type=='mag_spec':
        
        if fftSize == 512:
            f = 257
        elif fftSize == 256:
            f = 129
        elif fftSize == 1024:
            f = 513
        elif fftSize == 2048: 
            f = 1025
    else:
        concatenate=False
        if concatenate:
            f = 80      # when two types of features are concatenated (eg CQCC+SCMC)
        else:
            f = 40      # just the delta+acceleration (40 dimensional)    
                        
    weight_list = list()
    activation_list = list()
    bias_list = list()
    
    if activation=='mfm':
        fc_input= 13*17*8   #f*8   #6448 #1*257*64 = 16448
        in_conv2 = 8
        in_conv3 = 8
        in_conv4 = 8
        in_fc2 = 128
        in_fc3 = 128
        in_outputLayer = 128
        
    else:
        fc_input= 13*17*16   #f*16  #32896 # 1*257*128
        in_conv2 = 16
        in_conv3 = 16
        in_conv4 = 16
        in_fc2 = 256
        in_fc3 = 256
        in_outputLayer = 256
        
    #flattened = tf.reshape(pool4, shape=[-1, 65*19*32])
    print('======================== CNN ARCHITECTURE ==============================\n')    
                               
    #Convolution layer1,2,3    
    conv1,w1,b1 = conv_layer(input_placeholder, [3,10,1,16], [16], [1,1,1,1],'conv1',padding,activation,init)
    weight_list.append(w1)
    bias_list.append(b1)    
    print('Conv1 ', conv1)
    pool1 = maxPool2x2(conv1, [1,2,2,1], [1,2,2,1])
    
    conv2,w2,b2 = conv_layer(pool1, [3,10,in_conv2,16], [16], [1,1,1,1],'conv2', padding,activation,init)
    weight_list.append(w2)
    bias_list.append(b2)    
    print('Conv2 ', conv2)
    pool2 = maxPool2x2(conv2, [1,2,2,1], [1,2,2,1])
    
    conv3,w3,b3 = conv_layer(pool2, [3,10,in_conv3,16], [16], [1,1,1,1],'conv3', padding,activation,init)
    weight_list.append(w3)
    bias_list.append(b3)    
    #print('Conv3 ', conv3)
    pool3 = maxPool2x2(conv3, [1,2,2,1], [1,2,2,1])
    print('pool3 shape: ', pool3)
    
    if input_type == 'cqt_spec':
        time_dim = 32
    else:
        time_dim = t*100    
        
    # Dropout on the huge input from Conv layer    
    flattened = tf.reshape(pool3, shape=[-1,fc_input])
    dropped_1 = drop_layer(flattened, keep_prob1, 'dropout1')
    
    # Fully connected layer 1 with 256 neurons but gets splitted into 128 due to MFM
    fc1,w4,b4, = fc_layer(dropped_1, fc_input, 256, 'FC_Layer1', activation)
    weight_list.append(w4)
    bias_list.append(b4)
    
    print('Shape of FC1 = ', fc1.shape)
    
    '''
    # Dropout followed by FC layer with 256 neurons but gets splitted into 128 due to MFM
    dropped_2 = drop_layer(fc1, keep_prob2, 'dropout2')        
    fc2,w5,b5, = fc_layer(dropped_2, in_fc2, 256, 'FC_Layer2', activation)
    weight_list.append(w5)
    bias_list.append(b5)
    
    print('Shape of FC2 = ', fc2.shape)

    # Dropout followed by FC layer with 256 neurons but gets splitted into 128 due to MFM
    dropped_3 = drop_layer(fc2, keep_prob2, 'dropout3')        
    fc3,w6,b6, = fc_layer(dropped_3, in_fc3, 256, 'FC_Layer3', activation)
    weight_list.append(w6)
    bias_list.append(b6)
    
    print('Shape of FC3 = ', fc3.shape)
    ''' 

    #Output layer: 2 neurons. One for genuine and one for spoof. Dropout applied first
    dropped_4 = drop_layer(fc1, keep_prob3, 'dropout4')
    output,w7,b7 = fc_layer(dropped_4, in_outputLayer, targets, 'Output_Layer', 'no-activation')  #get raw logits        
        
    weight_list.append(w7)
    bias_list.append(b7)            
                
    print('Output layer shape = ', output.shape)
    print('======================== CNN ARCHITECTURE ==============================\n')
    
    
    return fc1, output, weight_list, activation_list, bias_list
#---------------------------------------------------------------------------------------------------------------
    
