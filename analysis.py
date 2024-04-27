import os
import librosa
import math
import json
import numpy as np
import matplotlib.pyplot as plt


JSON_PATH = "analysis.json"
DATASET_PATH = "D:/datasets/reverbs/data/reverb-mono-5"
MODEL_NAME = "realistic"

def analyze_dataset(dataset_path, json_path, n_mfcc=20, n_mels=30, n_fft=2048):

    data = {
        "mfccs": np.empty((n_mfcc, 1)),
        "mels": np.empty((n_mels, 1))
    } 

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            audio, sr = librosa.load(file_path, sr=None, mono=True)
            audio = librosa.util.fix_length(audio, size=n_fft)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=n_fft+1)
            mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=n_fft+1)
            data["mfccs"] = np.hstack((data["mfccs"], mfccs))
            data["mels"] = np.hstack((data["mels"], mels))
    
    mfcc_mean = np.mean(data["mfccs"], axis=1)
    mfcc_max = np.max(data["mfccs"], axis=1)
    mfcc_min = np.min(data["mfccs"], axis=1)

    mel_mean = np.mean(data["mels"], axis=1)
    mel_max = np.max(data["mels"], axis=1)
    mel_min = np.min(data["mels"], axis=1)

    to_return = {
        "mfcc": {
            "mean": mfcc_mean,
            "max": mfcc_max,
            "min": mfcc_min
        },
        "mels": {
            "mean": mel_mean,
            "max": mel_max,
            "min": mel_min
        }
    }
    
    return to_return

if __name__ == "__main__":
    config = json.load(open("config.json", "r"))[MODEL_NAME]
    analysis =  analyze_dataset(DATASET_PATH, JSON_PATH, n_mfcc=config["n_mfcc"], n_fft=config["n_fft"])


    mean_ft = librosa.feature.inverse.mel_to_stft(np.reshape(analysis["mels"]["mean"], (-1,1)), sr=44100)
    mean_db_ft = librosa.amplitude_to_db(mean_ft, ref=np.max, top_db=25.0)

    max_ft = librosa.feature.inverse.mel_to_stft(np.reshape(analysis["mels"]["max"], (-1,1)), sr=44100)
    max_db_ft = librosa.amplitude_to_db(max_ft, ref=np.max, top_db=25.0)

    min_ft = librosa.feature.inverse.mel_to_stft(np.reshape(analysis["mels"]["min"], (-1,1)), sr=44100)
    min_db_ft = librosa.amplitude_to_db(min_ft, ref=np.max, top_db=25.0)

    freqs = librosa.fft_frequencies(n_fft=2048, sr=44100)
    print(np.min(freqs), np.max(freqs))
    print(librosa.mel_frequencies(n_mels=30,fmin=25, fmax=20000, htk=True))
    

    fig, (ax1,ax3, ax2) = plt.subplots(3, 1)

    ax1.semilogx(librosa.mel_frequencies(n_mels=30, fmax=22500), librosa.amplitude_to_db(np.sqrt(analysis["mels"]["mean"])), label="mels mean")
    # ax1.plot(analysis["mels"]["max"], label="mels max")
    # ax1.plot(analysis["mels"]["min"], label="mels min")
    ax1.legend()

    ax3.semilogx(freqs, mean_db_ft, label="mean")
    # ax3.semilogx(freqs, max_db_ft, label="max")
    # ax3.semilogx(freqs, min_db_ft, label="min")
    ax3.legend()

    ax2.plot(analysis["mfcc"]["mean"], label="mfcc mean")
    ax2.plot(analysis["mfcc"]["max"], label="mfcc max")
    ax2.plot(analysis["mfcc"]["min"], label="mfcc min")
    ax2.legend()


    plt.show()

    # with open("analysis.json", "w") as fp:
    #     json.dump(analysis.tolist(), fp, indent=4)
    
    print("Analysis complete. Results saved to analysis.json")