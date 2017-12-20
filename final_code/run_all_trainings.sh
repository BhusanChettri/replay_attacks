#!/bin/bash

echo 'Training 2 target CNN without regularizer'         # Running on GPU 0
python train_cnn_pindArch_noReg_2targets.py

echo 'Training 4 target CNN without regularizer'    # Running on GPU 0
python train_cnn_pindArch_noReg_4targets.py

#echo 'Training 2 target CNN with 50% dropout regularizer'
#python train_cnn_pindArch_withReg_4targets.py


# We are running above in hepworth 0
# For each we will run these on three types of inputs
# We also run last one on hepworth 1, later on remeber to change the name of the folder after completing the execution... !


#echo 'Training 2 target CNN with 50% dropout regularizer'    # Running on 1 gpu after correction of inputs (new method used)
#python train_cnn_pindArch_withReg_2targets.py
