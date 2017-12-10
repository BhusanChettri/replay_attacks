import os
import numpy as np

def makeDirectory(path):
    # Create directories for storing tensorflow events    
    if not os.path.exists(path):
        os.makedirs(path)
        
def writetoFile(scores,file):    
    outFile = open(file, 'w')
    for val in scores:    
        outFile.write("%2.4f\n" % val)
    outFile.close()        

def getFeatureFiles(filelist, headPath):
    base='/homes/bc305/myphd/stage2/deeplearning.experiment1/CNN1/cnn_replayAttacks/cnn_features/'
    usePath= base+headPath
    
    featFiles=list()
    with open(filelist, 'r') as f:
        for line in f:            
            name=os.path.basename(line.strip()).replace('wav', 'feat')
            featFiles.append(usePath+name)
        return featFiles
    
def loadFeatures_from_file(feat_files, N):
    data = np.loadtxt(feat_files[0])
    for i in range(1, N):
        feature = np.loadtxt(feat_files[i])
        data=np.vstack((data, feature))
    return data

def loadFeatures(filename):            
    print('Loading features')
    
    if(os.path.isfile(filename)):
        with np.load(filename) as f:
            features = f['features']            
        return features
    
    else:
        print('No parameters found')
