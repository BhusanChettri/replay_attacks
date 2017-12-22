import scipy.io
import numpy as np
from helper import makeDirectory

def convert_matlab_to_numpy(matFile, saveFile):    
    mat = scipy.io.loadmat(matFile)
    data = mat['features']
    
    features = list()
    for matrix in data:
        features.append(matrix[0])
    
    # save the file    
    np.savez_compressed(saveFile, features=features)
    #return np.asarray(new_data)


#featTypes=['IMFCC', 'LPCC', 'LFCC', 'RFCC', 'CQCC.60','MFCC'] #'SCMC'
featTypes=['CQCC.60','MFCC'] #'SCMC'
base='/homes/bc305/myphd/stage2/stage1_scripts/afterInterspeech/repeat/individual_systems/'
saveBase='/homes/bc305/myphd/stage2/deeplearning.experiment1/features/'

for feat in featTypes:
    savePath = saveBase+feat
    makeDirectory(savePath)
    
    print('Converting training features..')
    train=base+feat+ '/features/20ms/train.mat'
    convert_matlab_to_numpy(train, savePath+'/train')
    
    print('Converting dev featrues..')
    dev=base+feat+ '/features/20ms/dev.mat'
    convert_matlab_to_numpy(dev, savePath+'/dev')
    
    #test=base+feat+ '/features/20ms/eval.mat'
    #convert_matlab_to_numpy(test, savePath+'/eval')