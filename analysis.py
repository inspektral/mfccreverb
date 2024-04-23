import os
import librosa
import math
import json
import numpy as np
import matplotlib.pyplot as plt


JSON_PATH = "analysis.json"
DATASET_PATH = "D:/datasets/reverbs/data/reverb-mono-5"
MODEL_NAME = "realistic"

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048):

    data = np.empty((n_mfcc, 1))
    print("data shape: "+str(data.shape))

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            audio, sr = librosa.load(file_path, sr=None, mono=True)
            audio = librosa.util.fix_length(audio, size=n_fft)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=n_fft+1)
            data = np.hstack((data, mfccs))
    
    print(data.shape)
    mean = np.mean(data, axis=1)
    print(mean.shape)
    print(mean)

    max = np.max(data, axis=1)
    print(max.shape)
    print(max)

    min = np.min(data, axis=1)
    print(min.shape)
    print(min)  
    
    return mean, max, min

if __name__ == "__main__":
    config = json.load(open("config.json", "r"))[MODEL_NAME]
    mean, max, min =  save_mfcc(DATASET_PATH, JSON_PATH, n_mfcc=config["n_mfcc"], n_fft=config["n_fft"])

    with open("analysis.json", "w") as fp:
        json.dump({"mean": mean.tolist(), "max": max.tolist(), "min": min.tolist()}, fp, indent=4)
    
    plt.plot(mean, label="mean")
    plt.plot(max, label="max")
    plt.plot(min, label="min")

    plt.legend()

    plt.show()

