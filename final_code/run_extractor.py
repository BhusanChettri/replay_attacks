import extract_cnn_scores_and_features as extractor

model_path = '../models/birdsArch_max2000epochs/_cnnModel2_keepProb_0.6_0.50.5lr0.0003/'
init_type='xavier'
activation='elu'
normType = 'global'
normalise=True
architecture = 2
specType='mag_spec'
targets=2
fftSize=256
duration=1
padding=True
batch_size = 100

featType=['scores','bottleneck']

extractor.get_scores_and_features(model_path,batch_size,init_type,activation,normType,normalise,architecture,specType,
                                  targets,fftSize,duration,padding,featType) 
