import extract_cnn_scores_and_features as extractor

base='/homes/bc305/myphd/stage2/deeplearning.experiment1/CNN3/'
model_path = base + '/models_augmented/model1_max100epochs_16batch/keep_0.1_0.2_0.3_mag_spec/'

init_type='xavier'
activation='mfm'
normType = 'global'
normalise=True
architecture = 1   #2        # 2 is bulbul, 5 is Russian
specType='mag_spec'
targets=2
fftSize=512
duration=1
padding=True
batch_size = 100

augment=True 

featType=['scores']   #,'bottleneck']   # we just want the CNN predictions here !

extractor.get_scores_and_features(model_path,batch_size,init_type,activation,normType,normalise,architecture,specType,
                                  targets,fftSize,duration,padding,featType,augment) 


