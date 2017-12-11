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

# Input mel-spectrogram of 1x1000X80, 1000 is time and 80 frequency units
#Conv1: 16, 3x3 filter no zero padding: 16X998X79
#Pool1: 3x3 filter, 16X332X26
#Conv2: 16, 3x3 filter no zero padding: 16X330X24
#Pool2: 3x3 filter, 16X110X8
#Conv3: 16, 3x1 filter no zero padding: 16X108X8
#Pool3: 3x1 filter, 16X36X8
#Conv4: 16, 3x1 filter no zero padding: 16X34X8
#Pool4: 3x1 filter, 16X11X8
#Fc1: 256
#FC2: 32
#FC3: 2  (they use 1)

def cnnModel1(trainSize,input_placeholder,activation,init,targets,fftSize,padding, keep_prob1, keep_prob2, keep_prob3):
    # Replicating the Bulbul architecture of Thomas grill. Note that they use mel-spectrogram
    # We are using power spectrogram at the moment
    
    print('FFT size used in this run is: ', fftSize)
    
    f=512  # lets take 512 as default    
    time_dim = trainSize * 100
    
    if fftSize == 512:
        f = 257
    elif fftSize == 256:
        f = 129
    elif fftSize == 1024:
        f = 513
    elif fftSize == 2048:
        f = 1025   
    
    weight_list = list()
    activation_list = list()
    bias_list = list()
            
    #if trainSize == '1sec':
    #    time_dim=100
    if activation=='mfm':
        if fftSize==512:
            fc_input=464   #2*29*8
        elif fftSize == 256:
            fc_input = 240
    else:
        if fftSize == 512:
            fc_input=928   # 2*29*16
        elif fftSize == 256:
            fc_input = 480
                           
    if activation=='mfm':
        in_conv2 = 8
        in_conv3 = 8
        in_conv4 = 8
        in_fc2 = 128        
        in_outputLayer = 16
    else:
        in_conv2 = 16
        in_conv3 = 16
        in_conv4 = 16
        in_fc2 = 256        
        in_outputLayer = 32
                       
    #Convolution layer1
    conv1,w1,b1 = conv_layer(input_placeholder, [3,3,1,16], [16], [1,1,1,1],'conv1',padding,activation,init)
    weight_list.append(w1)
    bias_list.append(b1)
    print('Conv1 ', conv1)

    pool1 = maxPool2x2(conv1, [1,3,3,1], [1,3,3,1])
    print('Pool1 layer shape = ', pool1)
    
    #in_conv2=
    conv2,w2,b2 = conv_layer(pool1, [3,3,in_conv2,16], [16], [1,1,1,1],'conv2',padding,activation,init)
    weight_list.append(w2)
    bias_list.append(b2)
    print('Conv2 ', conv2)

    pool2 = maxPool2x2(conv2, [1,3,3,1], [1,3,3,1])
    print('Pool2 layer shape = ', pool2)

    #in_conv3=
    conv3,w3,b3 = conv_layer(pool2, [3,1,in_conv3,16], [16], [1,1,1,1],'conv3',padding,activation,init)
    weight_list.append(w3)
    bias_list.append(b3)
    print('Conv3 ', conv3)

    pool3 = maxPool2x2(conv3, [1,3,1,1], [1,3,1,1])
    print('Pool3 layer shape = ', pool3)
    
    #in_conv4=
    conv4,w4,b4 = conv_layer(pool3, [3,1,in_conv4,16], [16], [1,1,1,1],'conv4',padding,activation,init)
    weight_list.append(w4)
    bias_list.append(b4)
    print('Conv4 ', conv4)
    
    pool4 = maxPool2x2(conv4, [1,3,1,1], [1,3,1,1])
    print('Pool4 layer shape = ', pool4)
    
    # Dropout on the huge input from Conv layer    
    flattened = tf.reshape(pool4, shape=[-1,fc_input])
    dropped_1 = drop_layer(flattened, keep_prob1, 'dropout1')
    
    # Fully connected layer 1 with 256 neurons but gets splitted into 128 due to MFM
    fc1,w4,b4, = fc_layer(dropped_1, fc_input, 256, 'FC_Layer1', activation)
    weight_list.append(w4)
    bias_list.append(b4)
    
    print('Shape of FC1 = ', fc1.shape)
        
    # Dropout followed by FC layer with 32 neurons but gets splitted into 128 due to MFM
    dropped_2 = drop_layer(fc1, keep_prob2, 'dropout2')        
    fc2,w5,b5, = fc_layer(dropped_2, in_fc2, 32, 'FC_Layer2', activation)
    print('Shape of FC1 = ', fc2.shape)
    weight_list.append(w5)
    bias_list.append(b5)
    
    #Output layer: 2 neurons. One for genuine and one for spoof. Dropout applied first
    dropped_3 = drop_layer(fc2, keep_prob3, 'dropout3')
    output,w6,b6 = fc_layer(dropped_3, in_outputLayer, targets, 'Output_Layer', 'no-activation')  #get raw logits
          
    weight_list.append(w6)
    bias_list.append(b6) 
                
    print('Output layer shape = ', output.shape)
    
    ## I want to train SVM classifier on 256 dim FC1 output and GMM on FC2 output
    #fc=list()
    #fc.append(fc1)
    #fc.append(fc2)
    
    return fc2, output, weight_list, activation_list, bias_list

#---------------------------------------------------------------------------------------------------------------


def cnnModel2(trainSize,input_placeholder,activation,init,targets,fftSize,padding, keep_prob1, keep_prob2, keep_prob3):
    # Replicating the Sparrow architecture of Thomas grill.
    # It uses only conv layers, no FC layer is used here.
    
    trainSize = str(trainSize) + 'sec'
    
    # Note: If this architecture works somehow. Then I will have to explore this deeply
    '''
             Filter  Depth, time, Frequency
    Input      -     1  x 100 x 127          #Assuming we are using 1sec, 256FFT. No padding.
    Conv1    (3x3)   32 x  98 x 127
    Conv2    (3x3)   32 x  96 x 125
    Pool1    (3x3)   32 x  32 x  42
    Conv3    (3x3)   32 x  30 x  40
    Conv4    (3x3)   32 x  28 x  38    
    Conv5    (3x20)  64 x  26 x  19        
    Pool2    (3x3)   64 x   9 x   7 
    Conv6    (9x1)  256 x   1 x   7
    Conv7    (1x1)   64 x   1 x   7
    Conv8    (1x1)   16 x   1 x   7
    O/p       2 Neurons                
    '''
    
    print('FFT size used in this run is: ', fftSize)
    f=512  # lets take 512 as default
    
    # Input = 100x257
    if fftSize == 512:
        f = 257        
    elif fftSize == 256:
        f = 129
    elif fftSize == 1024:
        f = 513
    elif fftSize == 2048:
        f = 1025   
    
    weight_list = list()
    activation_list = list()
    bias_list = list()
    
    if trainSize == '1sec':
        time_dim=100
        if activation=='mfm':
            if fftSize==512:
                fc_input= 168  #1*21*8
            elif fftSize == 256:
                fc_input = 56  #1*7*8
        else:
            if fftSize == 512:
                fc_input=336 #1*21*16
            elif fftSize == 256:
                fc_input =112   #1*7*16
                
    if activation == 'mfm':
        in_conv2 = 16
        in_conv3 = 16
        in_conv4 = 16
        in_conv5 = 16        
        in_conv6 = 32        
        in_conv7 = 128                
        in_conv8 = 32                
        
    else:
        in_conv2 = 32
        in_conv3 = 32
        in_conv4 = 32
        in_conv5 = 32                
        in_conv6 = 64        
        in_conv7 = 256                
        in_conv8 = 64     
        
    freq_inConv5 = 20   #chosen from their paper. Lets see the impact
    time_inConv6=9
                                           
    #Convolution layer1
    conv1,w1,b1 = conv_layer(input_placeholder, [3,3,1,32], [32], [1,1,1,1],'conv1',padding,activation,init)
    weight_list.append(w1)
    bias_list.append(b1)
    print('Conv1 ', conv1)
    
    conv2,w2,b2 = conv_layer(conv1, [3,3,in_conv2,32], [32], [1,1,1,1],'conv2',padding,activation,init)
    weight_list.append(w2)
    bias_list.append(b2)
    print('Conv2 ', conv2)

    pool1 = maxPool2x2(conv2, [1,3,3,1], [1,3,3,1])
    print('Pool1 layer shape = ', pool1)
    
    conv3,w3,b3 = conv_layer(pool1, [3,3,in_conv3,32], [32], [1,1,1,1],'conv3',padding,activation,init)
    weight_list.append(w3)
    bias_list.append(b3)
    print('Conv3 ', conv3)
    
    conv4,w4,b4 = conv_layer(conv3, [3,3,in_conv4,32], [32], [1,1,1,1],'conv4',padding,activation,init)
    weight_list.append(w4)
    bias_list.append(b4)
    print('Conv4 ', conv4)        
    
    conv5,w5,b5 = conv_layer(conv4, [3,freq_inConv5,in_conv5,64], [64], [1,1,1,1],'conv5',padding,activation,init)
    weight_list.append(w5)
    bias_list.append(b5)
    print('Conv5 ', conv5)
    
    pool2 = maxPool2x2(conv5,[1,3,3,1], [1,3,3,1])
    print('Pool2 layer shape = ', pool2)
        
    conv6,w6,b6 = conv_layer(pool2, [time_inConv6,1,in_conv6,256],[256],[1,1,1,1],'conv6',padding,activation,init)
    weight_list.append(w6)
    bias_list.append(b6)
    print('Conv6 ', conv6)

    conv7,w7,b7 = conv_layer(conv6, [1,1,in_conv7,64],[64],[1,1,1,1],'conv7',padding,activation,init)
    weight_list.append(w7)
    bias_list.append(b7)
    print('Conv7 ', conv7)
    
    conv8,w8,b8 = conv_layer(conv7, [1,1,in_conv8,16],[16],[1,1,1,1],'conv8',padding,activation,init)
    weight_list.append(w8)
    bias_list.append(b8)
    print('Conv8 ', conv8)
                
    # Dropout on the huge input from Conv layer    
    flattened = tf.reshape(conv8, shape=[-1,fc_input])        
    dropped_1 = drop_layer(flattened, keep_prob1, 'dropout1')

    # Output dense layer
    output,w9,b9 = fc_layer(dropped_1, fc_input, targets, 'Output_Layer', 'no-activation')  #get raw logits            
    weight_list.append(w9)
    bias_list.append(b9) 
                
    print('Output layer shape = ', output.shape)
    
    return flattened,output,weight_list,activation_list,bias_list


