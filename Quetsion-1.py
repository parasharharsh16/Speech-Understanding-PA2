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
        print(f"Processing model {model_path}")
        # Load the model
        model = load_models(model_path, model)
        model.to(device)
        # Set the model to evaluation mode
        model.eval()
        # Loop through the dataset
        for data in loaded_data:
            sample1, sample2, label = data["audio_1"], data["audio_2"], data["label"]
            score1 = model(sample1.to(device))
            score2 = model(sample2.to(device))
            score = F.cosine_similarity(score1, score2).item()

            if label == 1:
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)

        eer, eer_threshold = calculate_eer(genuine_scores, impostor_scores)
        output.append({"model": model_path, "eer": eer, "eer_threshold": eer_threshold})
    return output

def load_models(model_path, model_name):
    # Load the model architecture
    config_path = None
    
    if "hubert" in model_name:
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='fbank', config_path=config_path)    
        model.load_state_dict((torch.load(model_path))['model'], strict=False)
    elif "wav2vec2" in model_name:
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='fbank', config_path=config_path)    
        model.load_state_dict((torch.load(model_path))['model'], strict=False)
    elif "wavlm" in model_name:
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='fbank', config_path=config_path)  
        model.load_state_dict((torch.load(model_path))['model'], strict=False)
    else:
        raise ValueError("Unknown model type")
    # Set the model to evaluation mode
    return model

if __name__ == "__main__":
    eer_output = compare_models_on_voxceleb1h(config["models"], config["dataset"])
    output_df = pd.DataFrame(eer_output)
    print(f"EER calculation sfor all given 3 models on VoxCeleb1-H dataset with  dataset:")
    print(output_df)
    print("Completed!")