import torch
import torchaudio
import numpy as np
import pandas as pd

# Define config for models to loop on each of them
config = {
            "models": ["hubert_large_finetune.pth","wav2vec2_xlsr_finetune.pth","wavlm_large_finetune.pth"],
            "model_folder": "models",
            "dataset": "dataset/voxceleb1_dataset"
            }

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

if __name__ == "__main__":
    # Load the dataset
    dataset = ""
    genuine_scores = []
    impostor_scores = []

    for model in config["models"]:
        model_path = f"{config['model_folder']}/{model}"
        print(f"Processing model {model_path}")
        # Load the model
        model = torch.load(model_path)
        # Set the model to evaluation mode
        model.eval()
        # Loop through the dataset
        for i in range(len(dataset)):
            for j in range(i + 1, len(dataset)):
                sample1, sample2 = dataset[i], dataset[j]
                score = model(sample1, sample2).item()

                if dataset.labels[i] == dataset.labels[j]:
                    genuine_scores.append(score)
                else:
                    impostor_scores.append(score)

        eer, eer_threshold = calculate_eer(genuine_scores, impostor_scores)
        print("EER(%):", eer * 100)
        print("EER Threshold:", eer_threshold)
