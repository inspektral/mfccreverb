import librosa.display
import librosa
import matplotlib.pyplot as plt
import numpy as np

AUDIO_FILE = "./audio_from_mlp.wav"

if __name__ == "__main__":
    audio, sr = librosa.load(AUDIO_FILE, sr=44100)
    spectrum = librosa.stft(audio, n_fft=2048, hop_length=512)
    spectrumdb = librosa.amplitude_to_db(np.abs(spectrum), ref=np.max)
    plt.figure()
    librosa.display.specshow(spectrumdb, sr=sr, hop_length=512, x_axis='time', y_axis='log')
    plt.colorbar()
    plt.show()
