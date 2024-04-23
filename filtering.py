import librosa
import numpy as np
from scipy.signal import butter, lfilter, filtfilt
import soundfile as sf
import json

window_size = 2048
hop_size = 512

y, sr = librosa.load("noise.wav", sr=44100)
D = librosa.stft(y, n_fft=window_size, hop_length=hop_size,  window='hann')


n_mfcc = 50  # Replace with the actual number of MFCCs you used

with open('generated.json', 'r') as f:
    data = json.load(f)
mfccs = data
print(mfccs[0])

n_filters = 40

def design_filterbank(sr, n_filters, order=2):
  lowcut = np.linspace(1, sr // 2, n_filters + 1)[:-1]
  highcut = np.linspace(1, sr // 2, n_filters + 2)[2:]
  print(lowcut, highcut)
  filters = []
  for i in range(n_filters):
    b, a = butter(order, [lowcut[i] / sr, highcut[i] / sr], btype='bandpass')
    filters.append((b, a))
  return filters

filters = design_filterbank(sr, n_filters)

filtered_stft = np.zeros_like(D)
for i in range(D.shape[1]):
  window = D[:, i]

  mels = librosa.feature.inverse.mfcc_to_mel(np.array([mfccs[i]]).T, n_mels=n_filters)
  mels = np.sqrt(mels)
  filtered_window = np.zeros_like(window)
  for j, (b, a) in enumerate(filters):
    filtered_band = filtfilt(b, a, window)
    filtered_window += mels[j] * filtered_band
  filtered_stft[:, i] = filtered_window

filtered_noise = librosa.istft(filtered_stft, hop_length=hop_size, window='hann')

filtered_noise /= np.abs(filtered_noise).max()
sf.write("filtered_noise.wav", filtered_noise, sr, format='WAV', subtype='PCM_16')


