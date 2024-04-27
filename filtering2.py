import librosa
import numpy as np
from scipy.signal import butter, lfilter, filtfilt
import soundfile as sf
import json

window_size = 2048
n_fft = 2048
hop_size = 512

y, sr = librosa.load("noise.wav", sr=44100)
D = librosa.stft(y, n_fft=window_size, hop_length=hop_size,  window='hann')


n_mfcc = 50  # Replace with the actual number of MFCCs you used

with open('generated.json', 'r') as f:
    data = json.load(f)
mfccs = data
print(mfccs[0])

n_filters = 30

filtered_stft = np.zeros_like(D)
for i in range(D.shape[1]):
  window = D[:, i]

  mels = librosa.feature.inverse.mfcc_to_mel(np.array([mfccs[i]]).T, n_mels=n_filters)
  mels = np.sqrt(mels)

  mel_filters = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_filters, htk=True)
  mel_filters *= mels

  mel_spectrogram = np.dot(mel_filters, np.abs(window))
  mel_spectrogram = np.reshape(mel_spectrogram, (-1, 1))

  filtered_spectrogram = librosa.feature.inverse.mel_to_stft(mel_spectrogram, sr=sr, n_fft=n_fft)
  filtered_spectrogram = filtered_spectrogram.flatten()
  print(filtered_spectrogram.shape)
  print(filtered_stft[:,i].shape)
  filtered_stft[:, i] = filtered_spectrogram * np.exp(1j * np.angle(window))

filtered_noise = librosa.istft(filtered_stft, hop_length=hop_size, window='hann')

filtered_noise /= np.abs(filtered_noise).max()
sf.write("filtered_noise2.wav", filtered_noise, sr, format='WAV', subtype='PCM_16')


