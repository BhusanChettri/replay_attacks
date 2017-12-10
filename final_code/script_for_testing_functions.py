from network import init_weights
from network import bias_variable
import tensorflow as tf

import dataset
#from dataset import prepareData_and_labels111
import os
import audio
import numpy as np


def checkXavier():
    #This code is to test whether xavier init is working or not
    weights = init_weights([5,5,1,5], 'conv1', init_type='xavier')
    biases = bias_variable([5], 'conv1', init_type='xavier')
    
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))               
    init = tf.global_variables_initializer()       
    sess.run(init)
    
    w = sess.run(weights)
    print('Xavier Weights', w)
    b = sess.run(biases)
    print('Xavier biases', b)
    
def checkOrthogonal():
    # Check for orthogonal
    weights = init_weights([5,5,1,5], 'conv1', init_type='orthogonal')
    biases = bias_variable([5], 'conv1', init_type='orthogonal')
    
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))               
    init = tf.global_variables_initializer()       
    sess.run(init) 
    
    w = sess.run(weights)
    print('Orthogonal Weights', w)
    
    b = sess.run(biases)
    print('Orthogonal biases', b)
    

#from tensorflow.contrib.keras.layers import LeakyReLU as LRelu
#from tensorflow.contrib.keras.python.keras.layers import advanced_activations

def testArchitecture():
    # Test architecture
    from nn_architecture import cnnModel0
    print('Reset graph and create data and label placeholders..')
    tf.reset_default_graph()
    
    #test flag for batch norm
    tst = tf.placeholder(tf.bool)
    itr = tf.placeholder(tf.int32)
    num_classes=2
           
    #Placeholders for data and labels
    with tf.name_scope('input'):
        input_data  = tf.placeholder(tf.float32, [None, 307500], name='x_input')
        true_labels = tf.placeholder(tf.float32, [None, num_classes], name = 'y_input')            
    
    # Placeholders for droput probability
    keep_prob1 = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)
    act = tf.nn.elu
    a, model_prediction, network_weights = cnnModel0(input_data, act, 'xavier', keep_prob1, keep_prob2)
    
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))               
    init = tf.global_variables_initializer()       
    sess.run(init)



def test():        
    
    #Define parameters for computing spectrograms    
    fs       = 16000  #16khz
    fft_size = 2048   #128ms
    win_size = 2048   #128ms
    hop_size = 160    #10ms
    duration = 3      #3000ms    
    
    # Validation data: 25:25 (genuine and spoofed). Total 50 test files
    validation_list='/homes/bc305/myphd/datasets/ASVSpoof2017/filelists/dev_25:25.scp'
    validation_key='/homes/bc305/myphd/datasets/ASVSpoof2017/labels/dev_25:25.lab'
            
    dataList = validation_list
    dataLabels = validation_key
        
    #Load trained session and model parameters    
    model_path = 'models/'
    n_model = 2
    batch_size = 32
    num_classes=2
    mean_std_file='temp.mean_std'
    
    mode='training'
    savePath = 'test'   #../../spectrograms/3sec/
    
    normalise=True
    normType = 'global'
    
    #1. Compute spectrograms or load the existing one
    data,labels = dataset.spectrograms(dataList,dataLabels,savePath,fft_size,win_size,hop_size,duration)
    
    #2. If normalize=True normalize else dont
    if normalise:
        data=dataset.normalise_data(data,mean_std_file,normType,mode)
        print('Normalization done !!')
                                
    return data,labels

#data, labels = test()
#print(data.shape)
#print(len(data))


##### Test whether shuffle function is returning the correct labels and data
from dataset import iterate_minibatches

def checkShuffle():        
    
    #Define parameters for computing spectrograms    
    fs       = 16000  #16khz
    fft_size = 2048   #128ms
    win_size = 2048   #128ms
    hop_size = 160    #10ms
    duration = 3      #3000ms    
    
    # Validation data: 25:25 (genuine and spoofed). Total 50 test files
    validation_list='/homes/bc305/myphd/datasets/ASVSpoof2017/filelists/dev_25:25.scp'
    validation_key='/homes/bc305/myphd/datasets/ASVSpoof2017/labels/dev_25:25.lab'
        
    dataList = validation_list
    dataLabels = validation_key
        
    #Load trained session and model parameters    
    model_path = 'models/'
    n_model = 2
    batch_size = 5
    num_classes=2
    mean_std_file='mean_std.npz'
    normType = 'global'
    mode='training'
    savePath = 'temp/XXX'
        
    data,labels = dataset.spectrograms(dataList,dataLabels,savePath,fft_size,win_size,hop_size, duration)
    batch_generator=dataset.iterate_minibatches(data,labels, batch_size, shuffle=True)
    
    total_batches = int(len(data)/batch_size)   #+1
    print('Total batches = ', total_batches)
    for j in range(total_batches):
        data, labels = next(batch_generator)                       
        data, labels = dataset.reshape_minibatch(data, labels)       
        #print(len(data))
        #print(labels)
        print(data)
        
        
def checkSpectrogramCompute():
    
    #Define parameters for computing spectrograms    
    fs       = 16000  #16khz
    fft_size = 2048   #128ms
    win_size = 2048   #128ms
    hop_size = 160    #10ms
    duration = 3      #3000ms    
    
    # Validation data: 25:25 (genuine and spoofed). Total 50 test files
    validation_list='/homes/bc305/myphd/datasets/ASVSpoof2017/filelists/dev_25:25.scp'
    validation_key='/homes/bc305/myphd/datasets/ASVSpoof2017/labels/dev_25:25.lab'
        
    dataList = validation_list
    dataLabels = validation_key
        
    #Load trained session and model parameters    
    model_path = 'models/'
    n_model = 2
    batch_size = 5
    num_classes=2
    mean_std_file='mean_std.npz'
    normType = 'global'
    mode='training'
    savePath = 'temp/XXX'
    
    dataFile=validation_list
    with open(dataFile,'r') as f:
        for filename in f:
            spec=audio.compute_spectrogram(filename.strip(), fft_size=2048, win_size=2048, hop_size=160, duration=3)
            #print([spec==0])
            print(spec.shape)
                
        
#checkShuffle()        


#checkSpectrogramCompute()

checkXavier()