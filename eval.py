import numpy as np
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import math

MODEL_NAME = "lofi"  

model = tf.keras.models.load_model(MODEL_NAME+".h5")

def predict(data):
    data = np.array(data)
    data = np.expand_dims(data, axis=0)
    prediction = model.predict(data)
    return prediction

if __name__ == "__main__":
    input_data = json.load(open("test_input.json", "r"))["mfcc_test_3"]
    config = json.load(open("config.json", "r"))[MODEL_NAME]

    mfcc_to_generate = math.floor(config["sample_rate"]*config["duration"]/config["hop_length"])
    
    mfccs = np.array([input_data])
    for _ in range(mfcc_to_generate):
        prediction = predict(mfccs[-1])
        if np.any(np.isnan(prediction)) or np.any(np.isinf(prediction)):
            print("Invalid prediction:", prediction)
        mfccs = np.vstack([mfccs, prediction[0]])

    print("Generated MFCCs:", mfccs)
    print("Generated MFCCs shape:", mfccs.shape)
    #dump to json
    with open("generated.json", "w") as fp:
        json.dump(mfccs.tolist(), fp, indent=4)
    #convert to audio
    audio = librosa.feature.inverse.mfcc_to_audio(mfccs.T ,n_fft=config["n_fft"], hop_length=config["hop_length"], sr=config["sample_rate"])
    sf.write("audio_from_mlp.wav", audio, config["sample_rate"], format='WAV', subtype='PCM_16')

