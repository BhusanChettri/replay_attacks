%% This script is used to compute the Equal Error Rate on train, val and 
% test data set.
% Make sure to change the path for scorePath and saveFilename

clc;
clear all;

addpath(genpath('/import/c4dm-datasets/SpeakerRecognitionDatasets/eer/'));

%scorePath='../cnn_scores/usingBulbul/_cnnModel2_keepProb_0.6_0.5_0.5lr0.0003/'
%scorePath='../svm_scores/_cnnModel1_keepProb_0.1_0.2_0.2/usingTrain_svmPenalty_1e-05/'

scorePath = '../svm_scores/usingBulbul_keepProb_0.6_0.5_0.5lr0.0003/usingTrain_svmPenalty_1e-05/'

%%
disp('Computing EER using training set');
trainLabels=importdata('/import/c4dm-datasets/SpeakerRecognitionDatasets/ASVSpoof2017/labels/train.lab');
scores=importdata(strcat(scorePath,'/train_prediction.txt'));
eer_train = get_eer(scores,trainLabels);
disp('EER values on Training set');
eer_train

%%
disp('Computing EER using Dev set');
devLabels=importdata('/import/c4dm-datasets/SpeakerRecognitionDatasets/ASVSpoof2017/labels/dev.lab');
scores=importdata(strcat(scorePath,'/dev_prediction.txt'));
eer_dev = get_eer(scores,devLabels);
disp('EER values on Dev set');
eer_dev

%%


disp('Computing EER using Eval set');
evalLabels=importdata('/import/c4dm-datasets/SpeakerRecognitionDatasets/ASVSpoof2017/labels/eval_genFirstSpoof.lab');
scores=importdata(strcat(scorePath, '/eval_prediction.txt'));
eer_eval = get_eer(scores,evalLabels);
disp('EER values on Eval set');
eer_eval

%% Write the results to the file
%writeToFile(eer_train, saveFilename, 'On Training set');
%writeToFile(eer_dev, saveFilename, 'On Dev set');
%writeToFile(eer_eval, saveFilename, 'On Evaluation set');
