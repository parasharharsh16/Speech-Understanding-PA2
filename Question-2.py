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
from multiprocessing import Pool
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
def align_audio_data(audio1, audio2):
    audio1 = audio1.cpu()
    audio2 = audio2.cpu()
    if audio1.shape[0] > audio2.shape[0]:
        padding = torch.zeros(audio1.shape[0] - audio2.shape[0])
        audio2 = torch.cat((audio2, padding))
    elif audio2.shape[0] > audio1.shape[0]:
        padding = torch.zeros(audio2.shape[0] - audio1.shape[0])
        audio1 = torch.cat((audio1, padding))
    audio1 = audio1.to(device)
    audio2 = audio2.to(device)
    return audio1, audio2

def calculate_sisnri_and_sdrimprovement(groundtruth, separatedaudio,i):
    reference_audio_data = torch.tensor(groundtruth)
    separated_audio_data = torch.tensor(separatedaudio[0])
    reference_audio_data = reference_audio_data.to(device)
    separated_audio_data = separated_audio_data.to(device)
    # Pad the shorter tensor with zeros
    reference_audio_data,separated_audio_data = align_audio_data(reference_audio_data, separated_audio_data)

    # Calculate SISNRi
    sisnr_calculator = ScaleInvariantSignalNoiseRatio().to(device)
    sisnr_score = sisnr_calculator(separated_audio_data, reference_audio_data)
    sisnri_score = (sisnr_score - 1) / (sisnr_score + 1)
    
    # Calculate SDR improvement
    sdr_calculator = SignalDistortionRatio().to(device)
    sdr_score = sdr_calculator(separated_audio_data, reference_audio_data)
    sdr_improvement_score = sdr_score
    return sisnri_score, sdr_improvement_score


def eval_model(model, testloader):
    model.eval()
    model.to(device)
    sisnri =[]
    sdr = []
    total = len(testloader.dataset)
    count = 1
    for data in testloader.dataset:
        print(f"Evaluating {count}/{total}")
        # mix = data[1]['mixture_ID']
        # source1 = data[1]['source_1_path']
        # mix_audio,_ = torchaudio.load(f"storage_dir/Libri2Mix/wav8k/max/test/mix_both/{mix}.wav")
        # source1_audio_data, _= librosa.load(f"storage_dir/LibriSpeech/{source1}", sr=8000)
        mix_audio = data['mixture']
        source1_audio_data = data['source1']
        mix_audio = mix_audio.to(device)#[torch.tensor(ma).to(device) for ma in mix_audio]
        est_sources = model(mix_audio)
        model_output =est_sources[:, :, 0].detach().cpu()
        sisnri_score, sdr_improvement_score = calculate_sisnri_and_sdrimprovement(source1_audio_data[0], model_output,count)
        sisnri.append(sisnri_score.cpu().numpy())
        sdr.append(sdr_improvement_score.cpu().numpy())
        count += 1
    return sisnri, sdr


def load_audio_data(audio_path):
    audio_data, _ = torchaudio.load(audio_path)
    return audio_data
def load_data(i, dataframe):
    mix = dataframe.iloc[i]['mixture_ID']
    source1 = dataframe.iloc[i]['source_1_path']
    mix_audio = load_audio_data(f"storage_dir/Libri2Mix/wav8k/max/test/mix_both/{mix}.wav")
    source1_audio_data = load_audio_data(f"storage_dir/LibriSpeech/{source1}")
    return {'mixture': mix_audio, 'source1': source1_audio_data}

def prepare_data(dataframe):
    with Pool() as p:
        list_of_data = p.starmap(load_data, [(i, dataframe) for i in range(len(dataframe))])
    return list_of_data
def finetune_model(model, trainloader):
    model.train()
    for param in model.parameters():
        param.requires_grad = False
    
    final_layer = list(model.modules())[-2]
    for param in final_layer.parameters():
        param.requires_grad = True

    model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = torch.nn.MSELoss()
    for epoch in range(10):
        for data in trainloader.dataset:
            # mix = data[1]['mixture_ID']
            # source1 = data[1]['source_1_path']
            # mix_audio,_ = torchaudio.load(f"storage_dir/Libri2Mix/wav8k/max/test/mix_both/{mix}.wav")
            # source1_audio_data, _= librosa.load(f"storage_dir/LibriSpeech/{source1}", sr=8000)
            mix_audio = data['mixture']
            source1_audio_data = data['source1']
            mix_audio = mix_audio.to(device)
            est_sources = model(mix_audio)
            model_output =est_sources[:, :, 0]
            source1_audio_data, model_output = align_audio_data(torch.tensor(source1_audio_data[0]).to(device), model_output[0])
            optimizer.zero_grad()
            loss = criterion(model_output, torch.tensor(source1_audio_data))
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch}")
    return model
def plot_eval_results(sisnri_scores,sdrimprovement_scores,state,plot_path):
    # Create histograms for SISNRi and SDRi scores
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    mean_sisnri = round(np.mean(sisnri_scores),4)
    mean_sdr = round(np.mean(sdrimprovement_scores),4)
    # Assume x_values_1 and x_values_2 are your x-coordinates for the scatter plots
    x_values_1 = range(len(sisnri_scores))
    x_values_2 = range(len(sdrimprovement_scores))
    fig.suptitle(f'{state} model evaluation, Mean SISNRi: '+str(mean_sisnri)+', Mean SDR: '+str(mean_sdr) , fontsize=16)
    axs[0].scatter(x_values_1, sisnri_scores, color='skyblue', edgecolor='black')
    axs[0].set_title('Scatter Plot of SISNRi Scores')
    axs[0].set_xlabel('Index')
    axs[0].set_ylabel('SISNRi')

    axs[1].scatter(x_values_2, sdrimprovement_scores, color='lightcoral', edgecolor='black')
    axs[1].set_title('Scatter Plot of SDR Improvement Scores')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('SDR Improvement')

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

print("Preparing dataset...")
dataset = librimix_dataset('metadata/Libri2Mix/libri2mix_test-clean.csv')

dataset = dataset[:len(dataset)]
#dataset = dataset[:200]
dataset = prepare_data(dataset)
train, test = train_test_split(dataset, test_size=0.3)
train_loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=True)
print("Loading pre-trained model...")
model = separator.from_hparams(source="speechbrain/sepformer-whamr", savedir='pretrained_models/sepformer-whamr',run_opts={"device":"cuda"})
print("Evaluating Pre-Trained Model...")
sisnri, sdr = eval_model(model, test_loader)
plot_eval_results(sisnri, sdr,'Pretrained','question-2 results/eval_results_pre_trained.png')

print("Finetuning model...")
model = finetune_model(model, train_loader)
torch.save(model,'finetuned_model.pth')
#model = torch.load('finetuned_model.pth')

print("Evaluating Finetuned Model...")
sisnri, sdr = eval_model(model, test_loader)
plot_eval_results(sisnri, sdr,'Finetuned','question-2 results/eval_results_finetuned.png')
