# GMM supporting functions.

import numpy as np
import scipy.io as sio
from sklearn import mixture
from helper import writetoFile
from helper import makeDirectory
import pickle

     
def trainGMM(gen_train_data, spoof_train_data, mixtures, gPath, sPath, init):
    #init='kmeans' or 'random'        
    makeDirectory(gPath)    
    makeDirectory(sPath)
    
    for component in mixtures:
        print('Training GMM for genuine using %d GMM with diagonal cov and kmeans initialization' % component)
        gmmGen = mixture.GaussianMixture(n_components=component, 
                                         covariance_type='diag', max_iter=100, init_params=init, verbose=2)    #using maximum 10 EM iterations dint help
        gmmGen.fit(gen_train_data)
        
        # Train GMM for Spoof data
        print('Training GMM for spoof using %d GMM with diagonal cov and kmeans initialization' % component)
        gmmSpoof = mixture.GaussianMixture(n_components=component, 
                                           covariance_type='diag', init_params = 'kmeans', verbose=2)    
        gmmSpoof.fit(spoof_train_data)
        
        gModelName = 'genuine_model_' + str(component) + '.p'
        sModelName = 'spoof_model_' + str(component) + '.p'
        
        # Save the models using pickle
        pickle.dump(gmmGen, open(gPath+gModelName,'wb'))       
        pickle.dump(gmmSpoof, open(sPath+sModelName, 'wb'))

# These two functions are implemented by Erfan
def log_thr(data,thr=1e-6,log_base=None):
    if log_base:
        return np.log(np.maximum(data,thr))/np.log(log_base)
    else:
        return np.log(np.maximum(data,thr))
    

def gmm_process(x,weight_vec,mean_mat,cov_mat,output="log_like"):
    # Covariance matrix (cov_mat) is assumed to be diagonal!
    if x.ndim == 1:
        x = x.reshape(1,-1)
    M,D_model = mean_mat.shape
    N,D_obs = x.shape # N: number of frames, D is dimesnion of feature (observation)
    
    if D_model != D_obs:
        raise ValueError("Dimensions of data (`D_obs`) and model (`D_model`) do not match!")
    else:
        D = D_model

    log_const_term  = 0.5*(D*np.log(2*np.pi)+np.sum(log_thr(cov_mat,1e-10),axis=1))

    loglike_x_m = np.empty((N,M))
    
    for n in range(N):
        loglike_x_m[n,:] = -0.5*np.sum(((x[n,:]-mean_mat)**2)/cov_mat,axis=1)-log_const_term # log[p(x|m)]

    joint_x_m = np.exp(loglike_x_m+np.log(weight_vec)) # p(x,m), N x M

    if output in ["log_like","log-like"]:
        return np.log(np.sum(joint_x_m,axis=1)) # sum{p(x|m)p(x)} over all m

    elif output in ["post","posterior"]:
        return joint_x_m/np.sum(joint_x_m,axis=1).reshape(-1,1)
    
def computeScores(testData,genGmm, spoofGmm):
    scores = list()
    
    for i in range(len(testData)):            
        data = testData[i]
               
        llk_genuine = np.mean(gmm_process(data, genGmm.weights_, genGmm.means_, genGmm.covariances_, 'log_like'))
        llk_spoof = np.mean(gmm_process(data, spoofGmm.weights_, spoofGmm.means_, spoofGmm.covariances_, 'log_like'))
        scores.append(llk_genuine - llk_spoof);
    return scores

    
def scoreTestFile(components, testFeatures, genModelPath, spoofModelPath, outPath):
    print('Scoring now...')
    for comp in components:
        #Load spoof and genuine GMM
        gModelName=genModelPath+'/genuine_model_' + str(comp) + '.p'
        sModelName=spoofModelPath+'/spoof_model_' + str(comp) + '.p'
        gmmGen = pickle.load(open(gModelName, 'rb'))
        gmmSpoof = pickle.load(open(sModelName, 'rb'))
        
        #Compute the scores now
        scores = computeScores(testFeatures, gmmGen, gmmSpoof)
        
        #Write scores to a file
        fileName = outPath+'/using_'+str(comp)+'_components.txt'
        writetoFile(scores, fileName)

