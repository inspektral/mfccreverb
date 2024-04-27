import os
import librosa
import math
import json
import numpy as np


JSON_PATH = "data_mels_db.json"
DATASET_PATH = "D:/datasets/reverbs/data/reverb-mono-5"
MODEL_NAME = "mel_hifi"

def save_mfcc(dataset_path, json_path, samples, n_mfcc=13, n_fft=2048, hop_length=512):

    data = {
        "mfccs": [],
    }

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            audio, sr = librosa.load(file_path, sr=None, mono=True)
            audio = librosa.util.fix_length(audio, size=samples)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfccs = mfccs.T

            data["mfccs"].append(mfccs.tolist())

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


def save_melspectrogram(DATASET_PATH, JSON_PATH, samples, n_mels=30, n_fft=2048, hop_length=512):
    data = {
        "mels": [],
    }

    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            audio, sr = librosa.load(file_path, sr=None, mono=True)
            audio = librosa.util.fix_length(audio, size=samples)
            mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
            mels = mels.T

            data["mels"].append(mels.tolist())

    with open(JSON_PATH, "w") as fp:
        json.dump(data, fp, indent=4)

def save_log_melspectrogram(DATASET_PATH, JSON_PATH, samples, n_mels=30, n_fft=2048, hop_length=512):
    data = {
        "mels": [],
    }

    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            audio, sr = librosa.load(file_path, sr=None, mono=True)
            audio = librosa.util.fix_length(audio, size=samples)
            mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
            mels = librosa.power_to_db(mels, ref=np.max)
            mels = mels.T

            data["mels"].append(mels.tolist())

    with open(JSON_PATH, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    config = json.load(open("config.json", "r"))[MODEL_NAME]
    save_log_melspectrogram(DATASET_PATH, JSON_PATH, samples=config["sample_rate"]*config["duration"], n_mels=config["n_mels"], n_fft=config["n_fft"], hop_length=config["hop_length"])
