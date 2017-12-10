%% This script is used to compute the Equal Error Rate on train, val and 
% test data set.
% Make sure to change the path for scorePath and saveFilename

clc;
clear all;

addpath(genpath('/import/c4dm-datasets/SpeakerRecognitionDatasets/eer/'));
mixtures = [1,2,3,4,6,8,16,32,64];

saveFilename='../eer/usingBulbul_keepProb_0.6_0.5_0.5lr0.0003';
scorePath='../gmm_scores/usingBulbul/_cnnModel2_keepProb_0.6_0.5_0.5lr0.0003/usingTrain/'


%%
disp('Computing EER using training set');
trainLabels=importdata('/import/c4dm-datasets/SpeakerRecognitionDatasets/ASVSpoof2017/labels/train.lab');
tbase=strcat(scorePath,'/onTrain/using_');
eerListTrain = eer(mixtures, tbase,trainLabels);
disp('EER values on Training set');
eerListTrain'

%%
disp('Computing EER using Dev set');
devLabels=importdata('/import/c4dm-datasets/SpeakerRecognitionDatasets/ASVSpoof2017/labels/dev.lab');
dbase=strcat(scorePath,'/onDev/using_');
eerListDev = eer(mixtures, dbase,devLabels);
disp('EER values on Dev set');
eerListDev'

%%
disp('Computing EER using Eval set');
evalLabels =importdata('/import/c4dm-datasets/SpeakerRecognitionDatasets/ASVSpoof2017/labels/eval_genFirstSpoof.lab');
ebase=strcat(scorePath,'/onEval/using_');
eerListEval = eer(mixtures,ebase,evalLabels);
disp('EER values on Eval set');
eerListEval'


%% Write the results to the file
writeToFile(mixtures, saveFilename, 'Mixture Components');
writeToFile(eerListTrain, saveFilename, 'On Training set');
writeToFile(eerListDev, saveFilename, 'On Dev set');
writeToFile(eerListEval, saveFilename, 'On Evaluation set');


%% ======================== REdo this with loop
%{
saveFilename='../eer/result_usingCNN1_devData';                     % path to save the eer
scorePath='../gmm_scores/usingCNN1_mfm_0.2Drop_usingDev/'

%%
disp('Computing EER using training set');
trainLabels=importdata('/import/c4dm-datasets/SpeakerRecognitionDatasets/ASVSpoof2017/labels/train.lab');
tbase=strcat(scorePath,'/onTrain/using_');
eerListTrain = eer(mixtures, tbase,trainLabels);
disp('EER values on Training set');
eerListTrain'

%%
disp('Computing EER using Dev set');
devLabels=importdata('/import/c4dm-datasets/SpeakerRecognitionDatasets/ASVSpoof2017/labels/dev.lab');
dbase=strcat(scorePath,'/onDev/using_');
eerListDev = eer(mixtures, dbase,devLabels);
disp('EER values on Dev set');
eerListDev'

%%
disp('Computing EER using Eval set');
#evalLabels=importdata('/import/c4dm-datasets/SpeakerRecognitionDatasets/ASVSpoof2017/labels/eval_asper_originalTrials.lab');
evalLabels =importdata('/import/c4dm-datasets/SpeakerRecognitionDatasets/ASVSpoof2017/labels/eval_genuineFirstspoof.lab');
ebase=strcat(scorePath,'/onEval/using_');
eerListEval = eer(mixtures,ebase,evalLabels);
disp('EER values on Eval set');
eerListEval'


%% Write the results to the file
writeToFile(mixtures, saveFilename, 'Mixture Components');
writeToFile(eerListTrain, saveFilename, 'On Training set');
writeToFile(eerListDev, saveFilename, 'On Dev set');
writeToFile(eerListEval, saveFilename, 'On Evaluation set');

%}
