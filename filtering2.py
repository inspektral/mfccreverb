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

  # Compute the Mel spectrum from the MFCCs
  mels = librosa.feature.inverse.mfcc_to_mel(np.array([mfccs[i]]).T, n_mels=n_filters)
  mels = np.sqrt(mels)

  # Create a Mel filter bank based on the Mel spectrum
  mel_filters = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_filters, htk=True)
  mel_filters *= mels

  # Apply the Mel filter bank to the magnitude spectrum of the window
  mel_spectrogram = np.dot(mel_filters, np.abs(window))
  mel_spectrogram = np.reshape(mel_spectrogram, (-1, 1))

  # Convert the Mel spectrogram back to a power spectrogram
  filtered_spectrogram = librosa.feature.inverse.mel_to_stft(mel_spectrogram, sr=sr, n_fft=n_fft)
  filtered_spectrogram = filtered_spectrogram.flatten()
  print(filtered_spectrogram.shape)
  print(filtered_stft[:,i].shape)
  # Convert the power spectrogram back to a complex STFT by using the phase of the original STFT
  filtered_stft[:, i] = filtered_spectrogram * np.exp(1j * np.angle(window))

filtered_noise = librosa.istft(filtered_stft, hop_length=hop_size, window='hann')

filtered_noise /= np.abs(filtered_noise).max()
sf.write("filtered_noise.wav", filtered_noise, sr, format='WAV', subtype='PCM_16')


