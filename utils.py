import os
import librosa
import math
import json
import numpy as np
import matplotlib.pyplot as plt

def mfcc_to_json(file_path, json_path, n_mfcc=20, n_fft=2048, hop_length=512 , sr=44100):
    data = np.empty((n_mfcc, 1))
    print("data shape: "+str(data.shape))

    audio, sr = librosa.load(file_path, sr=sr, mono=True)
    audio = librosa.util.fix_length(audio, size=sr*5)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    print(mfccs.shape)
    
    
    with open(json_path, "w") as fp:
        json.dump(mfccs.T.tolist(), fp, indent=4)


def mel_to_mfcc(mels, n_mfcc=20, n_fft=2048, hop_length=512 , sr=44100):
    mfccs = np.empty((n_mfcc, 1))
    print("mfccs shape: "+str(mfccs.shape))
    for mel in mels:
        mel = np.array(mel)
        mel = np.expand_dims(mel, axis=0)
        mfcc = librosa.feature.inverse.mel_to_mfcc(mel.T, n_mfcc=n_mfcc, n_fft=n_fft, sr=sr, hop_length=hop_length)
        mfccs = np.hstack((mfccs, mfcc))
    print(mfccs.shape)
    return mfccs