def get_averaged_score(scoreFile, labelFile):
    # scoreFile that holds the raw scores
    # path of spectrogram/feature used to extract the labels, passed as labelFile
    
    print('debugging ...')   
    import numpy as np
    from itertools import groupby
    from dataset import load_data
    import os
    
    head, tail = os.path.split(scoreFile)
    outFile = head+'/'+tail.split('.')[0]+'_new.txt'
    
    # Put the scores from the file into a list
    with open(scoreFile) as f:
        new_scores=[line.strip() for line in f]

    # Load the spects/input to get the labels that was used and put it in group     
    d,l=load_data(labelFile)
    labels=[label.split(' ')[0] for label in l]
    new_labs = [list(j) for i, j in groupby(labels)]

    print(new_labs[0])
    print(new_scores[0:5])    
    print(len(new_labs))
    print(len(new_scores))

    
    n=0
    scoreList=list()
    for i in range(len(new_labs)):    
        temp=list()    
        for j in range(len(new_labs[i])):  
            temp.append(new_scores[n])
            n+=1
        
        scoreList.append(temp)
    
    print(len(scoreList))    
    avg_scores=list()
    
    with open(outFile, 'w') as f:
        for values in scoreList:
            a = np.mean(np.asarray(values, np.float))
            f.write(str(a)+'\n')
            avg_scores.append(a)
    
    print(len(avg_scores))
    #return avg_scores,scoreList    
#===============================================================================================

base='/homes/bc305/myphd/stage2/deeplearning.experiment1/'
scores=base+'/CNN3/models_augmented/model1_max100epochs_16batch/keep_0.1_0.2_0.3_mag_spec/predictions/'
specs=base+'/spectrograms_augmented/1sec_shift/mag_spec/512FFT/1sec/'


print('Combining dev scores...')
devScore=scores+'dev_prediction.txt'
devSpecs=specs+'dev/'
get_averaged_score(devScore, devSpecs)

print('Combining train scores..')
trainScore=scores+'train_prediction.txt'
trainSpecs=specs+'train/'
get_averaged_score(trainScore, trainSpecs)


#### Some issue with eval one !! TO check and cross verify this asap
print('Combining eval scores')
evalScore=scores+'eval_prediction.txt'
evalSpecs=specs+'eval/'
get_averaged_score(evalScore, evalSpecs)
