import numpy as np
import os

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from helper import makeDirectory
from helper import loadFeatures
from helper import writetoFile

def get_data_in_matrix(data):
    return np.asarray([matrix for dataList in data for matrices in dataList for matrix in matrices])

def train_svm(featPath, savePath, penalty, train_svm_on):
    # This svm is trained on default parameters of scikit learn
    
    gen_train_count = 1507
    gen_dev_count = 760
    spf_dev_count = 950
    gen_eval_count = 1298
    spf_eval_count = 12008
    
    train_features = get_data_in_matrix(loadFeatures(featPath + '/train/features.npz'))
    dev_features = get_data_in_matrix(loadFeatures(featPath+'/dev/features.npz'))
    eval_features=get_data_in_matrix(loadFeatures(featPath+'/eval/features.npz'))
    
    print('Total features in training set=', len(train_features))
    
    # Get genuine and spoofed labels from the training dataset   
    gen_train_labels = np.ones(gen_train_count)    # +1 labels from genuine class
    spoof_train_labels = 0-gen_train_labels        # -1 labels from spoofed class
    all_train_labels = np.hstack((gen_train_labels, spoof_train_labels))
    
    # Get genuine and spoofed labels from the Development/Validation dataset
    gen_dev_labels = np.ones(gen_dev_count)        # +1 labels from genuine class
    spoof_dev_labels = 0-np.ones(spf_dev_count)    # -1 labels from spoofed class
    all_dev_labels = np.hstack((gen_dev_labels, spoof_dev_labels))
       
    # Get genuine and spoofed examples from the Evaluation set
    gen_eval_labels = np.ones(gen_eval_count)          # +1 labels from genuine class
    spoof_eval_labels = 0-np.ones(spf_eval_count)      # -1 labels from spoofed class
    all_eval_labels = np.hstack((gen_eval_labels, spoof_eval_labels))
    
    train_data=None
    train_labels=None
    
    if train_svm_on == 'train':
        print('Training SVM using only Training features')
        train_data = train_features        
        train_labels = all_train_labels
        
    elif train_svm_on == 'dev':
        print('Training SVM on Validation/Dev features')
        train_data = dev_features        
        train_labels = all_dev_labels
        
    elif train_svm_on == 'eval':
        print('Training SVM on Evaluation features')
        train_data = eval_features        
        train_labels = all_eval_labels
        
    elif train_svm_on == 'pooled':
        print('Training SVM on train+dev features')
        train_data = np.vstack((train_features, dev_features))
        train_labels = np.hstack((all_train_labels, all_dev_labels))
       
    elif train_svm_on == 'all':
        print('Training SVM on entire dataset')
        print('Make sure eval features are extracted using correct label files !')
        train_data=np.vstack((np.vstack((train_features, dev_features)), eval_features))
        train_labels=np.vstack((np.vstack((all_train_labels, all_dev_labels)), all_eval_labels))
        
    print('Total features in Final training set=',len(train_data))
    print('Total labels in Final training set=',len(train_labels))
    
    # Standardise the data first before training SVM
    scaler = StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    
    # Train the SVM models
    print('Training the SVM models..')
    svm = LinearSVC(C=penalty,random_state=0)
    svm.fit(train_data, train_labels)
    
    #genModelPath = savePath+'/svm/'     
    
    print('Finished training the SVM !!')
    print('\n\n *******************************************************************************')
    
    return svm,scaler
    
#-----------------------------------------------------------------------------------------------------------

def test_svm(svm, scaler, test_feature_file, outputFile):    
    test_data = get_data_in_matrix(loadFeatures(test_feature_file))
    
    #Standardize the test data using scaler
    test_data = scaler.transform(test_data)
    scores = svm.decision_function(test_data)
    writetoFile(scores,outputFile)    
    
#-----------------------------------------------------------------------------------------------------------
def train_all_SVMs(penalty_list, featPath, savePath, folderName, train_on):
    for penalty in penalty_list:
        
        #1. Training on Train features
        print('Training SVM ...')
        savePath=savePath+folderName+str(penalty)
        
        makeDirectory(savePath)        
        svm,scaler = train_svm(featPath, savePath, penalty, train_on)
        
        # Testing
        print('Testing SVM ...')
        test_svm(svm,scaler, featPath + '/train/features.npz', savePath+'/train_prediction.txt')
        test_svm(svm,scaler, featPath + '/dev/features.npz', savePath+'/dev_prediction.txt')
        test_svm(svm,scaler, featPath + '/eval/features.npz', savePath+'/eval_prediction.txt')
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------

penalty_list = [0.00001]  #[0.1, 0.01, 0.001, 0.0001, 0.00001]

featPath = '../cnn_features/usingBulbul/_cnnModel2_keepProb_0.6_0.5_0.5lr0.0003_forSVM/'
savePath='../svm_scores/usingBulbul_keepProb_0.6_0.5_0.5lr0.0003/'



# Using Training Data
folderName='usingTrain_svmPenalty_'
train_all_SVMs(penalty_list, featPath, savePath, folderName, 'train')

# Using Dev data
#folderName='usingDev_svmPenalty_'
#train_all_SVMs(penalty_list, featPath, savePath, folderName, 'dev')

# Using Pooled data
#folderName='usingPooled_svmPenalty_'
#train_all_SVMs(penalty_list, featPath, savePath, folderName, 'pooled')


#### TODO: Centre the data. This is forgotton ! ------------- DONE on 27th Nov !
