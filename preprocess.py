import os
import librosa
import math
import json

SAMPLE_RATE = 22500
DURATION = 5
SAMPLES = SAMPLE_RATE * DURATION
JSON_PATH = "data.json"
DATASET_PATH = "D:/datasets/reverbs/data/reverb-mono-5"

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512):

    data = {
        "mfccs_input": [],
        "mfccs_target": []
    }

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            audio, sr = librosa.load(file_path, sr=None, mono=True)
            audio = librosa.util.fix_length(audio, size=SAMPLES)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfccs = mfccs.T

            data["mfccs_input"].append(mfccs.tolist()[:-1])
            data["mfccs_target"].append(mfccs.tolist()[1:])

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH)

