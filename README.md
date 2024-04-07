# Speech-Understanding-PA2
This repository contains code for Speech Understanding programming assignment-2 at IIT Jodhpur

##Setup for Question-1
--pip install -r requirements.txt
p.s. install Fire, Fairseq and S3prl

##Setup for Question-2
-- pip install torch torchaudio torchvision
-- pip install scikit-learn
-- pip install pandas

## Prepare dataset Question-1
-- Download Voxcelb1 dataset from torchaudio
-- Download Voxcelb1-H dataset matadata file
-- put them into dataset folder

-- Download Kalbeth dataset
-- put it in dataset folder

P.S. Please verify the dataset path in cofig dict in Question-1.py file
## Donwload models from the repo below
- https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification
models need to be downloaded are given below:
-- wav2vec2_xlsr_finetune.pth
-- hubert_large_finetune.pth
-- wavlm_large_finetune.pth

## Execute Question-1
-- python Question-1
### Output plot of Question-2 is in the folder `question-1 results`

## Prepare dataset Question-2
-- https://github.com/JorisCos/LibriMix.git
-- Delete folders other than Libri2Mix and Wham_noise
-- Delete files other than libri2mix_test-clean_info.csv, libri2mix_test-clean.csv
-- Execute the generate_librimix_changed.sh from current code (not cloned git)

## Execute Question-2
-- python Question-2

### Output plot of Question-2 is in the folder `question-2 results`

This code is being build with the help of follwing git repositories
-- https://github.com/JorisCos/LibriMix
-- https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification
-- https://github.com/AI4Bharat/IndicSUPERB
-- https://huggingface.co/speechbrain/sepformer-whamr