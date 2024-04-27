import numpy as np
import librosa
import soundfile as sf

num_samples = 512
white_noise = np.random.normal(0, 1, num_samples)

filtered_noise, sample_rate = librosa.load('./filtered_noise_100.wav', sr=44100)

convolved_signal = np.convolve(white_noise, filtered_noise, mode='same')
convolved_signal = convolved_signal / np.max(np.abs(convolved_signal))

sf.write('./convolved_signal.wav', convolved_signal, sample_rate)