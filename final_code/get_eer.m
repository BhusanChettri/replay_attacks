function [EER] = get_eer(scores,ground_truth)

addpath(genpath('/import/c4dm-datasets/SpeakerRecognitionDatasets/eer/'));
[Pmiss,Pfa] = rocch(scores(strcmp(ground_truth,'genuine')),scores(strcmp(ground_truth,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100;        

end