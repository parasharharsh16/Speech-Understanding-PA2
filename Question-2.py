from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split
import librosa
import numpy as np
import torch
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio,SignalDistortionRatio
import matplotlib.pyplot as plt
device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
class librimix_dataset(Dataset):
    def __init__(self, metadata_file):
        self.file_path = metadata_file
        self.data = self.load_metadata()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        datarow = self.data.iloc[idx]
        return datarow
    def load_metadata(self):
        df = pd.read_csv(self.file_path)
        #drop unnecessary columns
        df = df[['mixture_ID','source_1_path','source_2_path']]
        return df

def calculate_sisnri_and_sdrimprovement(groundtruth, separatedaudio,i):
    reference_audio_data = torch.tensor(groundtruth)
    separated_audio_data = torch.tensor(separatedaudio[0])
    # Pad the shorter tensor with zeros
    if reference_audio_data.shape[0] > separated_audio_data.shape[0]:
        padding = torch.zeros(reference_audio_data.shape[0] - separated_audio_data.shape[0])
        separated_audio_data = torch.cat((separated_audio_data, padding))
    elif separated_audio_data.shape[0] > reference_audio_data.shape[0]:
        padding = torch.zeros(separated_audio_data.shape[0] - reference_audio_data.shape[0])
        reference_audio_data = torch.cat((reference_audio_data, padding))

    # Calculate SISNRi
    sisnr_calculator = ScaleInvariantSignalNoiseRatio()
    sisnr_score = sisnr_calculator(separated_audio_data, reference_audio_data)
    sisnri_score = (sisnr_score - 1) / (sisnr_score + 1)
    
    # Calculate SDR improvement
    sdr_calculator = SignalDistortionRatio()
    sdr_score = sdr_calculator(separated_audio_data, reference_audio_data)
    sdr_improvement_score = sdr_score
    print(f"count:{i}")
    return sisnri_score, sdr_improvement_score


def eval_model(model, testloader):
    model.eval()
    model.to(device)
    sisnri =[]
    sdr = []
    count = 0
    for data in testloader.dataset.iterrows():
        mix = data[1]['mixture_ID']
        source1 = data[1]['source_1_path']
        mix_audio,_ = torchaudio.load(f"storage_dir/Libri2Mix/wav8k/max/test/mix_both/{mix}.wav")
        source1_audio_data, _= librosa.load(f"storage_dir/LibriSpeech/{source1}", sr=8000)
    
        mix_audio = mix_audio.to(device)#[torch.tensor(ma).to(device) for ma in mix_audio]
        est_sources = model(mix_audio)
        model_output =est_sources[:, :, 0].detach().cpu()
        sisnri_score, sdr_improvement_score = calculate_sisnri_and_sdrimprovement(source1_audio_data, model_output,count)
        sisnri.append(sisnri_score)
        sdr.append(sdr_improvement_score)
        count += 1
    return sisnri, sdr


def plot_eval_results(sisnri_scores,sdrimprovement_scores):
    # Create histograms for SISNRi and SDRi scores
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Assume x_values_1 and x_values_2 are your x-coordinates for the scatter plots
    x_values_1 = range(len(sisnri_scores))
    x_values_2 = range(len(sdrimprovement_scores))

    axs[0].scatter(x_values_1, sisnri_scores, color='skyblue', edgecolor='black')
    axs[0].set_title('Scatter Plot of SISNRi Scores')
    axs[0].set_xlabel('Index')
    axs[0].set_ylabel('SISNRi')

    axs[1].scatter(x_values_2, sdrimprovement_scores, color='lightcoral', edgecolor='black')
    axs[1].set_title('Scatter Plot of SDR Improvement Scores')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('SDR Improvement')

    plt.tight_layout()
    plt.savefig('eval_results.png')
    plt.show()

dataset = librimix_dataset('metadata/Libri2Mix/libri2mix_test-clean.csv')
#take sample of the dataset
dataset = dataset[:100]
train, test = train_test_split(dataset, test_size=0.3)
train_loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=True)
model = separator.from_hparams(source="speechbrain/sepformer-whamr", savedir='pretrained_models/sepformer-whamr')
sisnri, sdr = eval_model(model, test_loader)
plot_eval_results(sisnri, sdr)