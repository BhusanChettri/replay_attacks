#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Network.py

This file holds various functions of a neural network within the tensorflow.
We may change it later on when implementing a class.
'''

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


def cnnModel0(trainSize, input_placeholder, activation, init, keep_prob1, keep_prob2): 
    ## We have added one FClayer of 256neurons
    
    network_weights = list()
    activationList = list()
    biasList = list()
    
    print('------------ Using cnnModel0 New Architecture -----------')                        
    #Convolution layer1    
    conv1,w1,b1 = conv_layer(input_placeholder, [5,5,1,32], [32], [1,1,1,1], 'conv1', act=activation, init_type=init)
    
    #print(conv1.shape)
    pool1 = maxPool2x2(conv1, [1,2,2,1], [1,2,2,1])
    print('pool1.shape = ', pool1)
    
    network_weights.append(w1)
    activationList.append(pool1)
    biasList.append(b1)
        
    #Convolution layer2
    conv2,w2,b2 = conv_layer(pool1, [3,3,32,32], [32], [1,1,1,1], 'conv2', act=activation, init_type=init)
    network_weights.append(w2)
    #print(conv2.shape)
    pool2 = maxPool2x2(conv2, [1,2,2,1], [1,2,2,1])
    print('pool2.shape = ', pool2)
    
    network_weights.append(w2)
    activationList.append(pool2)
    biasList.append(b2)
        
    #Convolution layer3
    conv3,w3,b3= conv_layer(pool2, [3,3,32,32], [32], [1,1,1,1], 'conv3', act=activation, init_type=init)
    network_weights.append(w3)
    #print(conv3.shape)
    pool3 = maxPool2x2(conv3, [1,2,2,1], [1,2,2,1])
    print('pool3.shape = ', pool3)
    
    network_weights.append(w3)
    activationList.append(pool3)
    biasList.append(b3)    
    
    #Convolution layer4
    conv4,w4,b4 = conv_layer(pool3, [3,3,32,32], [32], [1,1,1,1], 'conv4', act=activation, init_type=init)
    network_weights.append(w4)
    #print(conv4.shape)
    pool4 = maxPool2x2(conv4, [1,2,2,1], [1,2,2,1])
    print('pool4.shape = ', pool4)
    
    network_weights.append(w4)
    activationList.append(pool4)
    biasList.append(b4)

    #Fully connected layer1 with dropout
    flattened = tf.reshape(pool4, shape=[-1, 65*19*32])
    dropped_1 = drop_layer(flattened, keep_prob1, 'dropout1')      
    fc1,fcw1,b5 = fc_layer(dropped_1, 65*19*32, 256, 'FC_Layer1', act=activation, init_type=init)
    network_weights.append(fcw1)
                   
    #Fully connected layer2 with dropout
    dropped_2 = drop_layer(fc1, keep_prob2, 'dropout2')      
    fc2,fcw2,b6 = fc_layer(dropped_2, 256, 64, 'FC_Layer2', act=activation, init_type=init)    
        
    network_weights.append(fcw2)
    activationList.append(dropped_2)
    biasList.append(b6)
                    
    #Output layer: 2 neurons. One for genuine and one for spoof
    dropped_3 = drop_layer(fc2, keep_prob2, 'dropout3')      
    output,fcw3,b7 = fc_layer(dropped_3, 64, 2, 'Output_Layer', 'no-activation', init_type=init)  #get raw logits
    print('Final output layer shape= ', output.shape)
    
    network_weights.append(fcw3)
    activationList.append(output)
    biasList.append(b7)
    
    return dropped_2, output, network_weights, activationList, biasList

#---------------------------------------------------------------------------------------------------------------
def cnnModel1(trainSize,input_placeholder, activation, init, keep_prob1, keep_prob2):
    
    network_weights = list()
    network_weights = list()
    activationList = list()
    biasList = list()

    print('------------ Using cnnModel1 architecture ...')
        
    #Convolution layer1    
    conv1,w1,b1 = conv_layer(input_placeholder, [5,5,1,32], [32], [1,1,1,1], 'conv1', act=activation, init_type=init)    
    pool1 = maxPool2x2(conv1, [1,2,2,1], [1,2,2,1])
    print('pool1.shape = ', pool1)

    network_weights.append(w1)
    activationList.append(pool1)
    biasList.append(b1)
   
    
    #Convolution layer2
    conv2,w2,b2 = conv_layer(pool1, [3,3,32,48], [48], [1,1,1,1], 'conv2', act=activation, init_type=init)
    pool2 = maxPool2x2(conv2, [1,2,2,1], [1,2,2,1])
    print('pool2.shape = ', pool2)
    
    network_weights.append(w2)
    activationList.append(pool2)
    biasList.append(b2)

    
    #Convolution layer3
    conv3,w3,b3 = conv_layer(pool2, [3,3,48,64], [64], [1,1,1,1], 'conv3', act=activation, init_type=init)        
    pool3 = maxPool2x2(conv3, [1,2,2,1], [1,2,2,1])
    print('pool3.shape = ', pool3)
    
    network_weights.append(w3)
    activationList.append(pool3)
    biasList.append(b3)
    
    #Convolution layer4
    conv4,w4,b4 = conv_layer(pool3, [3,3,64,32], [32], [1,1,1,1], 'conv4', act=activation, init_type=init)    
    network_weights.append(w4)    
    pool4 = maxPool2x2(conv4, [1,2,2,1], [1,2,2,1])
    print('pool4.shape = ', pool4)
 
    network_weights.append(w4)
    activationList.append(pool4)
    biasList.append(b4)
    
    #Convolution layer5
    conv5,w5,b5= conv_layer(pool4, [3,3,32,32], [32], [1,1,1,1], 'conv5', act=activation, init_type=init)           
    pool5 = maxPool2x2(conv5, [1,2,2,1], [1,2,2,1])
    print('pool5.shape = ', pool5.shape)

    network_weights.append(w5)
    activationList.append(pool5)
    biasList.append(b5)
                
    #Fully connected layer1 with dropout
    flattened = tf.reshape(pool5, shape=[-1, 33*10*32])  #33*10*32=10560
    dropped_1 = drop_layer(flattened, keep_prob1, 'dropout1')             
    fc1,w6,b6, = fc_layer(dropped_1, 33*10*32, 256, 'FC_Layer1', activation)
    
    network_weights.append(w6)
    activationList.append(fc1)
    biasList.append(b6)    
    
    ## NOTE: 33*10*32=10560 which is huge.And we are using only 256 neurons in FC1 layer
    ## I think this will loose lot of information. May be we try 1024 0r more neurons to
    ## capture more information ???? Think on this architecture !
        
    #Fully connected layer2 with dropout
    dropped_2 = drop_layer(fc1, keep_prob2, 'dropout2')             
    fc2,w7,b7 = fc_layer(dropped_2, 256, 64, 'FC_Layer2', activation)
        
    network_weights.append(w7)
    activationList.append(fc2)
    biasList.append(b7)
    
    #Output layer: 2 neurons. One for genuine and one for spoof
    dropped_3 = drop_layer(fc2, keep_prob2, 'dropout3')             
    output,w8,b8 = fc_layer(dropped_3, 64, 2, 'Output_Layer', 'no-activation')  #get raw logits
    print(output.shape)

    network_weights.append(w8)
    activationList.append(output)
    biasList.append(b8) 
    
    return fc2, output, network_weights,activationList, biasList

#---------------------------------------------------------------------------------------------------------------

def cnnModel2(trainSize,input_placeholder, activation, init, keep_prob1, keep_prob2):
    
    network_weights = list()
    network_weights = list()
    activationList = list()
    biasList = list()
        
    time_dim=300    # default
    fc_input=5*17*32  # default input if time_dim=300 for this architecture:  #10*33*32=10560

    if trainSize == '4sec':
        time_dim = 400
        fc_input = 0       ## Check this before using this architecture if 4seconds input
    elif trainSize == '5sec':
        time_dim = 400
        fc_input = 0       ## Check this     

    print(' Using cnnModel2 architecture ...')
            
    #Convolution layer1    
    conv1,w1,b1 = conv_layer(input_placeholder, [5,5,1,32], [32], [1,1,1,1], 'conv1', act=activation, init_type=init)    
    pool1 = maxPool2x2(conv1, [1,2,2,1], [1,2,2,1])
    print('pool1.shape = ', pool1)

    network_weights.append(w1)
    activationList.append(pool1)
    biasList.append(b1)
      
    #Convolution layer2
    conv2,w2,b2 = conv_layer(pool1, [3,3,32,48], [48], [1,1,1,1], 'conv2', act=activation, init_type=init)
    pool2 = maxPool2x2(conv2, [1,2,2,1], [1,2,2,1])
    print('pool2.shape = ', pool2)
    
    network_weights.append(w2)
    activationList.append(pool2)
    biasList.append(b2)
    
    #Convolution layer3
    conv3,w3,b3 = conv_layer(pool2, [3,3,48,64], [64], [1,1,1,1], 'conv3', act=activation, init_type=init)        
    pool3 = maxPool2x2(conv3, [1,2,2,1], [1,2,2,1])
    print('pool3.shape = ', pool3)
    
    network_weights.append(w3)
    activationList.append(pool3)
    biasList.append(b3)
    
    #Convolution layer4
    conv4,w4,b4 = conv_layer(pool3, [3,3,64,64], [64], [1,1,1,1], 'conv4', act=activation, init_type=init)    
    network_weights.append(w4)    
    pool4 = maxPool2x2(conv4, [1,2,2,1], [1,2,2,1])
    print('pool4.shape = ', pool4)
 
    network_weights.append(w4)
    activationList.append(pool4)
    biasList.append(b4)
    
    #Convolution layer5
    conv5,w5,b5= conv_layer(pool4, [3,3,64,48], [48], [1,1,1,1], 'conv5', act=activation, init_type=init)           
    pool5 = maxPool2x2(conv5, [1,2,2,1], [1,2,2,1])
    print('pool5.shape = ', pool5.shape)

    network_weights.append(w5)
    activationList.append(pool5)
    biasList.append(b5)

    #Convolution layer6
    conv6,w6,b6= conv_layer(pool5, [3,3,48,32], [32], [1,1,1,1], 'conv6', act=activation, init_type=init)           
    pool6 = maxPool2x2(conv6, [1,2,2,1], [1,2,2,1])
    print('pool6.shape = ', pool6.shape)

    network_weights.append(w6)
    activationList.append(pool6)
    biasList.append(b6)
                
    #Fully connected layer1 with dropout
    flattened = tf.reshape(pool6, shape=[-1, fc_input])  #5*17*32=2720 for 300seconds input
    dropped_1 = drop_layer(flattened, keep_prob1, 'dropout1')    
    fc1,w7,b7, = fc_layer(dropped_1,fc_input,64, 'FC_Layer1', activation)        
  
    network_weights.append(w7)
    activationList.append(dropped_1)
    biasList.append(b7)
               
    #Output layer: 2 neurons. One for genuine and one for spoof
    dropped_2 = drop_layer(fc1, keep_prob2, 'dropout2')    
    output,w8,b8 = fc_layer(dropped_2, 64, 2, 'Output_Layer', 'no-activation')  #get raw logits
    print('Output layer = ', output.shape)

    network_weights.append(w8)
    activationList.append(output)
    biasList.append(b8) 
    
    return fc1, output, network_weights, activationList, biasList

#---------------------------------------------------------------------------------------------------------------

def cnnModel3(trainSize,input_placeholder, activation, init, keep_prob1, keep_prob2):
    
    network_weights = list()
    network_weights = list()
    activationList = list()
    biasList = list()

    time_dim=300    # default
    fc_input=10560  # default input if time_dim=300 for this architecture:  #10*33*32=10560

    if trainSize == '4sec':
        time_dim = 400
        fc_input = 13728       #13*33*32
    elif trainSize == '5sec':
        time_dim = 400
        fc_input = 16896

    print('------------ Using cnnModel3 architecture ...')
            
    #Convolution layer1    
    conv1,w1,b1 = conv_layer(input_placeholder, [5,5,1,32], [32], [1,1,1,1], 'conv1', act=activation, init_type=init)    
    pool1 = maxPool2x2(conv1, [1,2,2,1], [1,2,2,1])
    print('Conv1 layer after pooling: shape = ', pool1)

    # Russians do NIN after this. We put one conv layer. Convolution layer2 without Max pooling
    conv2,w2,b2 = conv_layer(pool1, [3,3,32,32], [32], [1,1,1,1], 'conv2', act=activation, init_type=init)
    print('Conv2 layer shape = ', conv2)
      
    #Convolution layer3
    conv3,w3,b3 = conv_layer(conv2, [3,3,32,48], [48], [1,1,1,1], 'conv3', act=activation, init_type=init)
    pool2 = maxPool2x2(conv3, [1,2,2,1], [1,2,2,1])
    print('After pooling in Conv3 layer, shape = ', pool2)
            
    #Convolution layer4
    conv4,w4,b4 = conv_layer(pool2, [3,3,48,48], [48], [1,1,1,1], 'conv4', act=activation, init_type=init)        
    print('Conv4 layer shape = ', conv4)

    #Convolution layer5
    conv5,w5,b5 = conv_layer(conv4, [3,3,48,64], [64], [1,1,1,1], 'conv5', act=activation, init_type=init)        
    pool3 = maxPool2x2(conv5, [1,2,2,1], [1,2,2,1])
    print('After pooling in Conv5 layer, shape = ', pool3)    
       
    #Convolution layer6
    conv6,w6,b6 = conv_layer(pool3, [3,3,64,64], [64], [1,1,1,1], 'conv6', act=activation, init_type=init)    
    print('Conv6 layer shape = ', conv6)
    
    #Convolution layer7
    conv7,w7,b7 = conv_layer(conv6, [3,3,64,32], [32], [1,1,1,1], 'conv7', act=activation, init_type=init)
    pool4 = maxPool2x2(conv7, [1,2,2,1], [1,2,2,1])
    print('After pooling in Conv7 layer, shape = ', pool4)    
    
    #Convolution layer8
    conv8,w8,b8 = conv_layer(pool4, [3,3,32,32], [32], [1,1,1,1], 'conv8', act=activation, init_type=init)
    print('Conv8 layer shape = ', conv8)

    #Convolution layer9
    conv9,w9,b9 = conv_layer(conv8, [3,3,32,32], [32], [1,1,1,1], 'conv9', act=activation, init_type=init)
    pool5 = maxPool2x2(conv9, [1,2,2,1], [1,2,2,1])
    print('After pooling in Conv9 layer, shape = ', pool5)
                
    #Fully connected layer1 with dropout
    flattened = tf.reshape(pool5, shape=[-1,fc_input])  #10*33*32=10560 if 3 seconds spectrogram as input
    dropped_1 = drop_layer(flattened, keep_prob1, 'dropout1')                            
    fc1,w7,b7, = fc_layer(dropped_1,fc_input, 64, 'FC_Layer1', activation)
                               
    #Output layer: 2 neurons. One for genuine and one for spoof
    dropped_2 = drop_layer(fc1, keep_prob2, 'dropout2')                        
    output,w8,b8 = fc_layer(dropped_2, 64, 2, 'Output_Layer', 'no-activation')  #get raw logits
    print(output.shape)
      
    return fc1, output, network_weights, activationList, biasList

#---------------------------------------------------------------------------------------------------------------
def cnnModel4(trainSize, input_placeholder, activation, init, keep_prob1, keep_prob2):
    ### To make it work for different trainSize the input will be of different dimension
    ## therefore the input to FC layer after reshape will be of different length
    ## Need to generalize it
    ## for now just make it work for 3 seconds.. TODO for tomorrow !!
    
    network_weights = list()
    network_weights = list()
    activationList = list()
    biasList = list()
    
    time_dim=300    # default
    fc_input=2720   # default input if time_dim=300 for this architecture

    if trainSize == '4sec':
        time_dim = 400
        fc_input = 3808         #7*17*32=3808
    elif trainSize == '5sec':
        time_dim = 400
        fc_input = 4352         #8*17*32=4352

    print('------------ Using cnnModel4 architecture ...')
    
    #Convolution layer1    
    conv1,w1,b1 = conv_layer(input_placeholder, [5,5,1,32], [32], [1,1,1,1], 'conv1', act=activation, init_type=init)    
    pool1 = maxPool2x2(conv1, [1,2,2,1], [1,2,2,1])      
    print('Conv1 layer after pooling: shape = ', pool1)

    # Russians do NIN after this. We put one conv layer. Convolution layer2 without Max pooling
    #Convolution layer2
    conv2,w2,b2 = conv_layer(pool1, [3,3,32,32], [32], [1,1,1,1], 'conv2', act=activation, init_type=init)
    print('Conv2 layer shape = ', conv2)
      
    #Convolution layer3
    conv3,w3,b3 = conv_layer(conv2, [3,3,32,48], [48], [1,1,1,1], 'conv3', act=activation, init_type=init)
    pool2 = maxPool2x2(conv3, [1,2,2,1], [1,2,2,1])
    print('After pooling in Conv3 layer, shape = ', pool2)
            
    #Convolution layer4
    conv4,w4,b4 = conv_layer(pool2, [3,3,48,48], [48], [1,1,1,1], 'conv4', act=activation, init_type=init)        
    print('Conv4 layer shape = ', conv4)

    #Convolution layer5
    conv5,w5,b5 = conv_layer(conv4, [3,3,48,64], [64], [1,1,1,1], 'conv5', act=activation, init_type=init)        
    pool3 = maxPool2x2(conv5, [1,2,2,1], [1,2,2,1])
    print('After pooling in Conv5 layer, shape = ', pool3)    
       
    #Convolution layer6
    conv6,w6,b6 = conv_layer(pool3, [3,3,64,64], [64], [1,1,1,1], 'conv6', act=activation, init_type=init)    
    print('Conv6 layer shape = ', conv6)
    
    #Convolution layer7
    conv7,w7,b7 = conv_layer(conv6, [3,3,64,32], [32], [1,1,1,1], 'conv7', act=activation, init_type=init)
    pool4 = maxPool2x2(conv7, [1,2,2,1], [1,2,2,1])
    print('After pooling in Conv7 layer, shape = ', pool4)    
    
    #Convolution layer8
    conv8,w8,b8 = conv_layer(pool4, [3,3,32,32], [32], [1,1,1,1], 'conv8', act=activation, init_type=init)
    print('Conv8 layer shape = ', conv8)

    #Convolution layer9
    conv9,w9,b9 = conv_layer(conv8, [3,3,32,32], [32], [1,1,1,1], 'conv9', act=activation, init_type=init)
    pool5 = maxPool2x2(conv9, [1,2,2,1], [1,2,2,1])
    print('After pooling in Conv9 layer, shape = ', pool5)

    #Convolution layer10
    conv10,w10,b10 = conv_layer(pool5, [3,3,32,32], [32], [1,1,1,1], 'conv10', act=activation, init_type=init)
    print('Conv10 layer shape = ', conv10)

    #Convolution layer11
    conv11,w11,b11 = conv_layer(conv10, [3,3,32,32], [32], [1,1,1,1], 'conv11', act=activation, init_type=init)
    pool6 = maxPool2x2(conv11, [1,2,2,1], [1,2,2,1])
    print('After pooling in Conv11 layer, shape = ', pool6)
            
    #Fully connected layer1 with dropout
    flattened = tf.reshape(pool6, shape=[-1, fc_input])     #fc_input = 5*17*32=2720 if 3seconds spectrogram
    dropped_1 = drop_layer(flattened, keep_prob1, 'dropout1') 
    print('Dropped1 shape = ', dropped_1)    
    fc1,w7,b7, = fc_layer(dropped_1,fc_input, 64, 'FC_Layer1', activation)
        
    # Output layer with dropout
    dropped_2 = drop_layer(fc1, keep_prob2, 'dropout2')     
    output,w8,b8 = fc_layer(dropped_2, 64, 2, 'Output_Layer', 'no-activation')  #get raw logits
    print('Output layer shape = ',output.shape)
      
    return fc1, output, network_weights, activationList, biasList

#---------------------------------------------------------------------------------------------------------------
def cnnModel5(trainSize,input_placeholder,activation,init,targets,fftSize,padding,keep_prob1,keep_prob2,keep_prob3):
    # This is the Exact replication of Russians Paper using Max-Feature-Map activation    
    #(trainSize,input_data, act,init_type,num_classes,fftSize,padding,keep_prob1,keep_prob2,keep_prob3)
    print('TrainSize input to architecture is: ', trainSize)
    trainSize = str(trainSize)+'sec'
    
        
    network_weights = list()
    network_weights = list()
    activationList = list()
    biasList = list()

    if trainSize == '1sec':
        time_dim=100
        if activation=='mfm':
            if fftSize==512:
                fc_input= 576   #4*9*16
            elif fftSize == 256:
                fc_input = 320       #4*5*16
        else:
            if fftSize == 512:
                fc_input= 1152       #4*9*32
            elif fftSize == 256:
                fc_input = 640      #4*5*32
                
    elif trainSize == '3sec':
        time_dim=300
        if activation=='mfm':
            fc_input=5280   #10*33*16
        else:
            fc_input=10560  #10*33*32
            
    elif trainSize == '4sec':
        time_dim = 400
        
        if activation=='mfm':
            if fftSize==2048:
                fc_input= 6864      #13*33*16
            elif fftSize == 512:
                fc_input = 0
            elif fftSize == 256:
                fc_input = 0
                
        else:
            if fftSize==2048:
                fc_input= 13728  # 13*33*32
            elif fftSize == 512:
                fc_input = 0
            elif fftSize == 256:
                fc_input = 0
                
    if activation=='mfm':
        in_conv2 = 16
        in_conv3 = 16
        in_conv4 = 24
        in_conv5 = 24
        in_conv6 = 32
        in_conv7 = 32
        in_conv8 = 16
        in_conv9 = 16
        in_outputLayer = 32        
    else:        
        in_conv2 = 32
        in_conv3 = 32
        in_conv4 = 48
        in_conv5 = 48
        in_conv6 = 64
        in_conv7 = 64
        in_conv8 = 32
        in_conv9 = 32
        in_outputLayer = 64

    print('------------ Using cnnModel5, Replicating Russian architecture with MFM Activation !!')
        
    #Convolution layer1         
    conv1,w1,b1 = conv_layer(input_placeholder,[5,5,1,32],[32],[1,1,1,1],'conv1',padding,activation,init)
    pool1 = maxPool2x2(conv1, [1,2,2,1], [1,2,2,1])
    print('Conv1 layer after pooling: shape = ', pool1)
    
    
    #NIN Layer
    conv2,w2,b2 = conv_layer(pool1, [1,1,in_conv2,32],[32],[1,1,1,1],'conv2',padding,activation,init)    
    print('Conv2 (NIN) layer shape = ', conv2)
            
    #Convolution layer3
    conv3,w3,b3 = conv_layer(conv2, [3,3,in_conv3,48], [48], [1,1,1,1],'conv3',padding,activation,init)
    pool2 = maxPool2x2(conv3, [1,2,2,1], [1,2,2,1])
    print('Conv3 after pooling, shape = ', pool2)               
    
    #NIN Layer    
    conv4,w4,b4 = conv_layer(pool2, [1,1,in_conv4,48], [48], [1,1,1,1], 'conv4',padding,activation,init)
    print('Conv4 layer shape = ', conv4)
        
    #Convolution layer5
    conv5,w5,b5 = conv_layer(conv4, [3,3,in_conv5,64], [64], [1,1,1,1], 'conv5',padding,activation,init)          
    pool3 = maxPool2x2(conv5, [1,2,2,1], [1,2,2,1])
    print('Conv5 layer after pooling, shape = ', pool3)    
               
    # NIN layer    
    conv6,w6,b6 = conv_layer(pool3, [1,1,in_conv6,64], [64], [1,1,1,1], 'conv6',padding,activation,init)    
    print('Conv6 layer shape = ', conv6)
            
    #Convolution layer7
    conv7,w7,b7 = conv_layer(conv6, [3,3,in_conv7,32], [32], [1,1,1,1], 'conv7',padding,activation,init)          
    pool4 = maxPool2x2(conv7, [1,2,2,1], [1,2,2,1])
    print('Conv7 after pooling , shape = ', pool4)
            
    # NIN Layer
    conv8,w8,b8 = conv_layer(pool4, [1,1,in_conv8,32], [32], [1,1,1,1], 'conv8',padding,activation,init)
    print('Conv8 layer shape = ', conv8)
                
    conv9,w9,b9 = conv_layer(conv8, [3,3,in_conv9,32], [32], [1,1,1,1], 'conv9',padding,activation,init)
    pool5 = maxPool2x2(conv9, [1,2,2,1], [1,2,2,1])
    print('Conv9 after pooling, shape = ', pool5)
                    
    # Dropout on the huge input from Conv layer
    flattened = tf.reshape(pool5, shape=[-1,fc_input])    
    dropped_1 = drop_layer(flattened, keep_prob1, 'dropout1')
    
    # Fully connected layer 1 with 64 neurons but gets splitted into 32 due to MFM
    fc1,w7,b7, = fc_layer(dropped_1, fc_input, 64, 'FC_Layer1', activation)
    print('Shape of FC1 = ', fc1.shape)        
    
    #Output layer: 2 neurons. One for genuine and one for spoof. Dropout applied first
    dropped_2 = drop_layer(fc1, keep_prob2, 'dropout2')                                                       
    output,w8,b8 = fc_layer(dropped_2, in_outputLayer, targets,'Output_Layer', 'no-activation')  #get raw logits
    print('Output layer shape = ', output.shape)
    
    print('Targets in arch is : ', targets)
    
    return fc1, output, network_weights, activationList, biasList

#---------------------------------------------------------------------------------------------------------------
