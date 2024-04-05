import torch
import torchaudio
import numpy as np
import pandas as pd
import torchaudio.models as models
import soundfile as sf
import torch
import fire
import torch.nn.functional as F
from torchaudio.transforms import Resample
from ecapa_tdnn import ECAPA_TDNN_SMALL
import multiprocessing as mp
from pydub import AudioSegment
import os
import random

# Define config for models to loop on each of them
config = {
            "models": ["wav2vec2_xlsr_finetune.pth","hubert_large_finetune.pth","wavlm_large_finetune.pth"],
            "model_folder": "models",
            "dataset": "dataset/archive/vox1_dev_wav/wav"
            }
device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
def calculate_eer(genuine_scores, impostor_scores):
    num_genuine_pairs = len(genuine_scores)
    num_impostor_pairs = len(impostor_scores)

    eer = 0.0
    min_difference = float('inf')
    eer_threshold = 0.0

    for threshold in np.arange(min(min(genuine_scores), min(impostor_scores)), max(max(genuine_scores), max(impostor_scores)), 0.001):
        far = sum(impostor_scores >= threshold) / num_impostor_pairs
        frr = sum(genuine_scores < threshold) / num_genuine_pairs
        difference = abs(far - frr)

        if difference < min_difference:
            min_difference = difference
            eer = (far + frr) / 2
            eer_threshold = threshold

    return eer, eer_threshold

# This function will read the VoxCeleb1-H txt file and return a pandas dataframe with the columns: label, audio_1_path, audio_2_path
def read_voxceleb1h_txt(file_path, percentage= 0.0001):
    df = pd.read_csv(file_path, sep=" ", header=None)
    df_with_headers = df.rename(columns={0: "label", 1: "audio_1_path", 2: "audio_2_path"})
    df_with_headers = df_with_headers.sample(frac=percentage, random_state=1)
    return df_with_headers

#This function will load the audio files from the dataframe and return data in the format: (audio_1, audio_2, label)
def worker(row, dataset_path):
    try:
        audio_1, sr = torchaudio.load(f"{dataset_path}/{row['audio_1_path']}")
        audio_2, sr = torchaudio.load(f"{dataset_path}/{row['audio_2_path']}")
        label = row['label']
        return {"audio_1": audio_1, "audio_2": audio_2, "label": label}
    except:
        return None

def load_audio_files(df, dataset_path):
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(worker, [(row, dataset_path) for _, row in df.iterrows()])
    return [result for result in results if result is not None]


def compare_models_on_voxceleb1h(models, dataset_path, percentage=0.0001):
    # Load the dataset
    dataset = read_voxceleb1h_txt("dataset/voxceleb-h.txt", percentage)
    loaded_data = load_audio_files(dataset, dataset_path)
    genuine_scores = []
    impostor_scores = []
    output = []
    for model in models:
        model_path = f"{config['model_folder']}/{model}"

        model = load_models(model_path)
        model.to(device)
        model.eval()
        # Loop through the dataset
        for data in loaded_data:
            sample1, sample2, label = data["audio_1"], data["audio_2"], data["label"]
            score = get_similarity_score(model,sample1.to(device), sample2.to(device))
            if label == 1:
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)

        eer, eer_threshold = calculate_eer(genuine_scores, impostor_scores)
        output.append({"model": model_path, "eer": eer, "eer_threshold": eer_threshold})
    return output

def load_models(model_path):
    # Load the model architecture
    config_path = None
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='fbank', config_path=config_path)    
    model.load_state_dict((torch.load(model_path))['model'], strict=False)
    return model

def prep_kalbath_dataset(dataset_path, lang1, lang2):
    data = []
    all_files = os.listdir(f"{dataset_path}/{lang1}/test_known/audio")
    all_files_diffent_lang = os.listdir(f"{dataset_path}/{lang2}/test_known/audio")
    #pick random files from lang2
    for file_path in os.listdir(f"{dataset_path}/{lang1}/test_known/audio"):
        lable_random = random.choice([0,1])
        file_path = f"{dataset_path}/{lang1}/test_known/audio/{file_path}"
        if lable_random == 0:
            file2 = f"{dataset_path}/{lang2}/test_known/audio/{random.choice(all_files_diffent_lang)}"
            data.append({"file_path_1": file_path, "file_path_2": file2, "label": 0})
        else:
            similar_file = f"{dataset_path}/{lang1}/test_known/audio/{random.choice(all_files)}"
            data.append({"file_path_1": file_path, "file_path_2": similar_file , "label": 1}) 
    return data

def readm4a_file(file_path):
    if not file_path.endswith(".wav"):
            audio = AudioSegment.from_file(file_path, format="m4a")
            wav_file_path = 'sample_file.wav'
            audio.export(wav_file_path, format="wav")
    else:
        wav_file_path = file_path
    
    data, samplerate = torchaudio.load(wav_file_path)
    resampler = torchaudio.transforms.Resample(orig_freq=samplerate, new_freq=16000)

    data = resampler(data)
    return data, 16000

def get_similarity_score(model, audio_1, audio_2):
    score1 = model(audio_1.to(device))
    score2 = model(audio_2.to(device))
    score = F.cosine_similarity(score1, score2).item()
    return score

def compare_models_on_Kathbath(models, dataset_path, percentage=0.0001):
    output = []
    for model in models:
        model_path = f"{config['model_folder']}/{model}"
        print(f"Processing model {model_path}")
        model = load_models(model_path)
        model.to(device)
        model.eval()
        for lang1, lang2 in [("hindi", "punjabi"), ("hindi", "tamil"), ("hindi", "sanskrit")]:
            dataset_combi = prep_kalbath_dataset(dataset_path, lang1, lang2)
            genuine_scores = []
            impostor_scores = []
            sample_data_len = int(len(dataset_combi) * percentage)
            sample_data = random.sample(dataset_combi, sample_data_len)
            for row in sample_data:
                audio_1, sr = readm4a_file(row['file_path_1'])
                audio_2, sr = readm4a_file(row['file_path_2'])
                label = row['label']
                audio_1 =(F.interpolate(audio_1.view(1,1,-1), size=(96001,), mode='linear', align_corners=False).view(1,-1))
                audio_2 =(F.interpolate(audio_2.view(1,1,-1), size=(96001,), mode='linear', align_corners=False).view(1,-1))
                score = get_similarity_score(model,torch.tensor(audio_1), torch.tensor(audio_2))
                if label == 1:
                    genuine_scores.append(score)
                else:
                    impostor_scores.append(score)
                
            eer, eer_threshold = calculate_eer(genuine_scores, impostor_scores)
            output.append({"model": model_path,"lang":f"{lang1}, {lang2}","eer": eer, "eer_threshold": eer_threshold})
    return output

if __name__ == "__main__":
    # eer_output = compare_models_on_voxceleb1h(config["models"], config["dataset"])
    # output_df = pd.DataFrame(eer_output)
    # print(f"EER calculation sfor all given 3 models on VoxCeleb1-H dataset with  dataset:")
    # print(output_df)

    print("Calculating EER for all given 3 models on Kathbath dataset")
    eer_output = compare_models_on_Kathbath(config["models"], "dataset/testkn_audio/kb_data_clean_m4a", 0.01)
    output_df = pd.DataFrame(eer_output)
    print(output_df)
    print("Completed!")