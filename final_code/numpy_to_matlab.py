import scipy.io as sio
import numpy as np
from helper import loadFeatures
import os

def makeDirectory(path):
    # Create directories for storing tensorflow events    
    if not os.path.exists(path):
        os.makedirs(path)

def convert_numpy_to_matlab(featPath, outFile):
    numpy_array =  loadFeatures(featPath)   
    sio.savemat(outFile, mdict={'data': numpy_array})

base='/homes/bc305/myphd/stage2/deeplearning.experiment1/CNN1/cnn_replayAttacks/cnn_features'

# Output matlab feature save path
savePath='features/using_4sec_cnnModel4_global_Normalization_dropout_0.8_0.5L2_reg/'
makeDirectory(savePath)

# Convert training features
print('Converting training features')
trainFeature = base+'/using_4sec_cnnModel4_global_Normalization_dropout_0.8_0.5L2_reg/train/features.npz'
convert_numpy_to_matlab(trainFeature, savePath+'train.mat')


# Convert development features
print('Converting dev features')
devFeature = base+'/using_4sec_cnnModel4_global_Normalization_dropout_0.8_0.5L2_reg/dev/features.npz'
convert_numpy_to_matlab(devFeature, savePath+'dev.mat')


# Convert evaluation features
print('Converting eval features')
evalFeature = base+'/using_4sec_cnnModel4_global_Normalization_dropout_0.8_0.5L2_reg/eval/features.npz'
convert_numpy_to_matlab(evalFeature, savePath+'eval.mat')

