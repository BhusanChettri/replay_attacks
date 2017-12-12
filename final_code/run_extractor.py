import extract_cnn_scores_and_features as extractor

base='/homes/bc305/myphd/stage2/deeplearning.experiment1/CNN3/models/'
#model_path = base + '/sparrowArch/sparrow_keep_0.4_lr0.00008_elu_fft256/'

model_path = base + '/rusCNN_max2000epochs/arch5_keep_0.3_0.4_4sec/'

# the code will extract the features and scores inside the model path folder itself !

init_type='xavier'
activation='mfm'
normType = 'global'
normalise=True
architecture = 5   #2        # 2 is bulbul, 5 is Russian
specType='mag_spec'
targets=2
fftSize=2048     #256
duration=4       #1
padding=True
batch_size = 10  #50 #100

featType=['scores']   #,'bottleneck']   # we just want the CNN predictions here !

extractor.get_scores_and_features(model_path,batch_size,init_type,activation,normType,normalise,architecture,specType,
                                  targets,fftSize,duration,padding,featType) 
