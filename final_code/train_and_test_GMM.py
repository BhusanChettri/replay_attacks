import numpy as np
import os

from gmm import trainGMM
from helper import getFeatureFiles
from helper import loadFeatures_from_file
from helper import makeDirectory
from helper import loadFeatures
from gmm import scoreTestFile

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def get_data_in_matrix(data):
    return np.asarray([matrix for dataList in data for matrices in dataList for matrix in matrices])

def applyPCA_decomposition(data, minDimension=20):
    pca = PCA().fit(data)
    
    cum_variance=list()
    variance=0
    featDimension=None
    for i in range(0, len(pca.explained_variance_ratio_)):
        variance += pca.explained_variance_ratio_[i]
        cum_variance.append(variance)
        if variance > 0.99:
            print('Variance is 99% with '+str(i+1)+' components')
            featDimension=i+1
            break
        #plt.plot(cum_variance)
        #plt.show()
    if featDimension<minDimension:
        print('Since featDimension is smaller than 20, we keep featDimension as 20')
        featDimension=minDimension
        
    return pca, featDimension

def train_GMM_Models(mixtures, featPath, savePath, init, train_gmm_on, apply_pca, keepDimension):
    train_features = loadFeatures(featPath + '/train/features.npz')
    dev_features = loadFeatures(featPath+'/dev/features.npz')
    eval_features=loadFeatures(featPath+'/eval/features.npz')
    
    if len(train_features) != 3014:
        train_features=get_data_in_matrix(train_features)
        dev_features=get_data_in_matrix(dev_features)
        eval_features=get_data_in_matrix(eval_features)        
    
    print('Total features in training set=', len(train_features))                        
    
    # Get genuine and spoofed examples from the training dataset
    gen_train   =train_features[0:1507]
    spoof_train =train_features[1507:]
    
    # Get genuine and spoofed examples from the Development/Validation dataset
    gen_dev   =dev_features[0:760]
    spoof_dev =dev_features[760:]
    
    # Get genuine and spoofed examples from the Test dataset
    gen_eval   =eval_features[0:1298]
    spoof_eval =eval_features[1298:]
    
    gen_train_data=None
    spoof_train_data=None        
    
    pca=None
    dim=None
    if train_gmm_on == 'train':
        print('Training GMM using only Training features')
        if apply_pca:
            pca,dim=applyPCA_decomposition(train_features, keepDimension)
            train_features=pca.transform(train_features)[:,0:dim]
            
        gen_train_data = train_features[0:1507]
        spoof_train_data = train_features[1507:]
        
    elif train_gmm_on == 'dev':
        print('Training GMM on Validation/Dev features')
        if apply_pca:
            pca,dim=applyPCA_decomposition(dev_features, keepDimension)
            dev_features=pca.transform(dev_features)[:,0:dim]
                
        gen_train_data=dev_features[0:760]
        spoof_train_data= dev_features[760:]
        
    elif train_gmm_on == 'eval':
        print('Training GMM on Evaluation features')
        if apply_pca:
            pca,dim=applyPCA_decomposition(eval_features, keepDimension)
            eval_features=pca.transform(eval_features)[:,0:dim]
                
        gen_train_data= eval_features[0:1298]
        spoof_train_data = eval_features[1298:]       
        
    elif train_gmm_on == 'pooled':
        print('Training GMM on combined/pooled features')
        if apply_pca:
            pooled_features = np.vstack((train_features,dev_features))
            pca,dim=applyPCA_decomposition(pooled_features, keepDimension)
            pooled_features=pca.transform(pooled_features)[:,0:dim]
            
            gen_train=pooled_features[0:1507,:] #first 1507 is train genuine
            spoof_train=pooled_features[1507:3014, :] #Next 1507 is train spoofed
            gen_dev = pooled_features[3014:3774, :]  #Next 760 is dev genuine
            spoof_dev = pooled_features[3774:]       #remaining 950 is dev spoofed
        
        gen_train_data=np.vstack((gen_train, gen_dev))
        spoof_train_data=np.vstack((spoof_train, spoof_dev))
        
    elif train_gmm_on == 'all':
        print('Training GMM on entire dataset')
        print('Make sure eval features are extracted using correct label files !')
        print('No PCA support is provided for this option !!')
        gen_train_data=np.vstack((np.vstack((gen_train, gen_dev)), gen_eval))
        spoof_train_data=np.vstack((np.vstack((spoof_train, spoof_dev)), spoof_eval)) 
        
    print('Total features in Final Genuine training set=',len(gen_train_data))
    print('Total features in Final Spoofed training set=',len(spoof_train_data))
    
    
    # Train the GMM models
    print('Training the GMM models..')
    genModelPath = savePath+'/genuine/'
    spoofModelPath = savePath+'/spoof/'
    
    trainGMM(gen_train_data, spoof_train_data, mixtures, genModelPath, spoofModelPath, init)
    print('Finished training the GMM models !!')
    print('\n\n *******************************************************************************')
    
    return pca,dim
    
#-----------------------------------------------------------------------------------------------------------

def test_GMM_Models(mixtures, gmmModelPath, scoreSavePath, test_feature_file,pca,dim):
    genModelPath = gmmModelPath+'/genuine/'
    spoofModelPath = gmmModelPath+'/spoof/'    
    makeDirectory(scoreSavePath)
    
    test_data = loadFeatures(test_feature_file)
    if len(test_data) != 3014 or len(test_data) != 1710 or len(test_data)!=13306:
        test_data=get_data_in_matrix(test_data)
    
    #apply pca
    if pca != None:
        test_data = pca.transform(test_data)[:,0:dim]     
        
    scoreTestFile(mixtures, test_data, genModelPath, spoofModelPath, scoreSavePath)
    
#-----------------------------------------------------------------------------------------------------------

def train_all_GMMs(mixtures, featPath, savePath, modelPath, train_on, apply_pca, keepDimension):
    init='random'
    
    # Train using the training features only
    pca,dim=train_GMM_Models(mixtures, featPath, modelPath, init, 'train', apply_pca, keepDimension)
    
    print('\n\nTesting GMM models on Training features')
    test_GMM_Models(mixtures, modelPath, savePath+'/onTrain/', featPath + '/train/features.npz',pca,dim)
    
    print('\n\nTesting GMM models on Validation/Dev features')
    test_GMM_Models(mixtures, modelPath, savePath+'/onDev/', featPath + '/dev/features.npz',pca,dim)
    
    print('\n\nTesting GMM models on Evaluation/Eval features')
    test_GMM_Models(mixtures, modelPath, savePath+'/onEval/', featPath + '/eval/features.npz',pca,dim)
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------


apply_pca=False
keepDimension=20 ##

mixtures = [1,2,3,4,6,8,16,32,64]
featPath='../cnn_features/usingBulbul/_cnnModel2_keepProb_0.6_0.5_0.5lr0.0003/'
modelPath = '../gmm_models/usingBulbul/_cnnModel2_keepProb_0.6_0.5_0.5lr0.0003/'     
savePath='../gmm_scores/usingBulbul/_cnnModel2_keepProb_0.6_0.5_0.5lr0.0003/'


# Using Training Data
train_on='train'
train_all_GMMs(mixtures, featPath, savePath+'usingTrain/', modelPath+'usingTrain/', train_on,
               apply_pca, keepDimension)

'''
# Using Dev Data
train_on='dev'
train_all_GMMs(mixtures, featPath, savePath+'usingDev/', modelPath+'usingDev/', train_on,
               apply_pca, keepDimension)
    
# Using Pooled Data
train_on='pooled'
train_all_GMMs(mixtures, featPath, savePath+'usingPooled/', modelPath+'usingPooled/', train_on,
               apply_pca, keepDimension)
'''