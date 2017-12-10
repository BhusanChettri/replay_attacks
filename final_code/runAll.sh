#!/bin/bash

# Make sure to specify proper paths to models in the respective files first !!

#-------------------------------------------------------------------------
#1. Train the CNN
#echo "Training CNN...."
#python train_cnn_on_trainData.py 
#echo "Finished Training the CNN"

#-------------------------------------------------------------------------
#2. Extract train+dev features from trained CNN
#echo "Extracting Train+Dev Features from the CNN...."
#python extract_cnn_features_train+dev.py

#-------------------------------------------------------------------------
#3. Extract test features from trained CNN
#echo "Extracting Test (Eval) Features from the CNN...."
#python extract_cnn_features_testset.py

#-------------------------------------------------------------------------
#4. Train GMM using the features extracted
#echo "Training the GMM models on extracted training features .."
# Make sure to change the necessary path before running this file 
#python train_and_test_GMM.py

#-------------------------------------------------------------------------
#5. To add script for matlab EER computation.
echo "Now computing the EER on training, development and evaluation dataset"
#bash evaluate_EER.sh
#matlab -nodisplay -nosplash -nojvm -r "run computeEER; quit;"

matlab -nodisplay -nosplash -nojvm -r "run eer_end_to_end; quit;"
