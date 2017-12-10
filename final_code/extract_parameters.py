#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
 Testing the model on evaluation data
'''

# Load standard modules
from __future__ import print_function
import sys
import os
import io
import numpy as np
import tensorflow as tf

from optparse import OptionParser
from network import init_weights
from network import bias_variable
from model import load_model
from helper import makeDirectory

# Load userdefined modules
import audio
import dataset
import model
import numpy as np
import nn_architecture
import nn_architecture_rus


def save_W_B(save_file, weights, biases):
    with open(save_file, 'w') as f:
        np.savez(save_file, W=weights, B=biases)
        

def load_W_B(filename):            
    print('Loading Weights and Biases')
    
    if(os.path.isfile(filename)):
        with np.load(filename) as f:
            W = f['W']
            B = f['B']
        return W, B
    
    else:
        print('No parameters found')               

def retrieveWeights(sess, filter_shape, layer_name):
    conv1_weights = None
    with tf.name_scope(layer_name):
            with tf.variable_scope('weights', reuse=True):
                weights = init_weights(filter_shape, layer_name)
                conv1_weights = sess.run(weights)
    return conv1_weights

def retrieveBiases(sess, shape, layer_name):
    bias_values = None
    with tf.name_scope(layer_name):
        with tf.variable_scope('biases', reuse=True):
            biases = bias_variable(shape, layer_name)
            bias_values = sess.run(biases)
            
    return bias_values

def access_learned_parameters(model_path, save_path, trainSize='1sec', activation='mfm', architecture=1, num_classes=2,
                             init_type='xavier'):
            
    tf.reset_default_graph()    

    if trainSize == '3sec':       
        input_data = tf.placeholder(tf.float32, [None, 300, 1025,1])  #make it 4d tensor
        true_labels = tf.placeholder(tf.float32, [None, num_classes], name = 'y_input')
    elif trainSize == '4sec':        
        input_data = tf.placeholder(tf.float32, [None, 400, 1025,1])  #make it 4d tensor
        true_labels = tf.placeholder(tf.float32, [None, num_classes], name = 'y_input')
    elif trainSize == '5sec':
        input_data = tf.placeholder(tf.float32, [None, 500, 1025,1])  #make it 4d tensor
        true_labels = tf.placeholder(tf.float32, [None, num_classes], name = 'y_input')
    elif trainSize == '1sec':
        input_data = tf.placeholder(tf.float32, [None, 100, 257,1]) # We use 512 fft
        true_labels = tf.placeholder(tf.float32, [None, num_classes], name = 'y_input')
            
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

    # Get model architecture that was used during training
    if architecture == 0:        
        featureExtractor, model_prediction,network_weights,activations,biases= nn_architecture.cnnModel0(trainSize, input_data, act,init_type,keep_prob1,keep_prob2,keep_prob3)
    elif architecture == 1:        
        featureExtractor, model_prediction,network_weights,activations,biases= nn_architecture.cnnModel1(trainSize,input_data, act,init_type,keep_prob1,keep_prob2,keep_prob3)
    elif architecture == 2:        
        featureExtractor, model_prediction,network_weights,activations,biases= nn_architecture.cnnModel2(trainSize,input_data, act,init_type,keep_prob1,keep_prob2,keep_prob3)
    elif architecture == 3:        
        featureExtractor, model_prediction,network_weights,activations,biases= nn_architecture.cnnModel3(trainSize,input_data, act,init_type,keep_prob1,keep_prob2,keep_prob3)
    elif architecture == 4:
        featureExtractor, model_prediction,network_weights,activations,biases= nn_architecture.cnnModel4(trainSize,input_data, act,init_type,keep_prob1,keep_prob2,keep_prob3)
    elif architecture == 5:
        featureExtractor, model_prediction,network_weights,activations,biases= nn_architecture_rus.cnnModel5(trainSize,input_data, act,init_type,keep_prob1,keep_prob2) # Russian arc use two drops
   
    #Load trained session and model parameters
    print('Model parameters loading..')
    sess, saver = load_model(model_path)
    weights,biases = sess.run([network_weights,biases])
    print('Total weights: ',len(weights))
    
    # Save all weight parameters
    save_W_B(save_path+'/all_weights_biases.npz', weights,biases)        

def get_weights_biases():
    model_path='../models/model1/using_1sec_cnnModel1_global_Normalization_dropout_0.1_0.4/'    
    save_path = '../model_parameters/pindrop_model1_keep0.1_0.2_0.4/'
    makeDirectory(save_path)        
    access_learned_parameters(model_path, save_path)
    
    
def retrieve_parameters(): 
    save_path = '../model_parameters/pindrop_model1_keep0.1_0.2_0.4/'          
    weights, biases = load_W_B(save_path+'/all_weights_biases.npz')
    
    for w,b in zip(weights, biases):
        print(w.shape, b.shape)
        
#get_weights_biases()
#retrieve_parameters()


#---------------- To be sent to Saumitra for loading parameters and starting analysis -----------------------#
'''
import os
import numpy as np

def retrieve_network_parameters(): 
    filename='../model_parameters/pindrop_model1_keep0.1_0.2_0.4/all_weights_biases.npz' # Change the path ! 
    
    with np.load(filename) as f:
        weights = f['W']
        biases = f['B']            
    
    for w,b in zip(weights, biases):
        print(w.shape, b.shape)
        
retrieve_network_parameters()
'''
#---------------- To be sent to Saumitra for loading parameters and starting analysis -----------------------#

            
