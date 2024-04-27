import numpy as np
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import math

MODEL_NAME = "mel_db_hifi"  

model = tf.keras.models.load_model(MODEL_NAME+".h5")

def predict(data):
    data = np.array(data)
    data = np.expand_dims(data, axis=0)
    prediction = model.predict(data)
    return prediction

def eval_mfcc_to_audio():
    input_data = json.load(open("test_input.json", "r"))["mfcc_noise2"]
    config = json.load(open("config.json", "r"))[MODEL_NAME]

    mfcc_to_generate = math.floor(config["sample_rate"]*config["duration"]/config["hop_length"])
    
    mfccs = np.array([input_data])
    for _ in range(mfcc_to_generate):
        prediction = predict(mfccs[-1])
        if np.any(np.isnan(prediction)) or np.any(np.isinf(prediction)):
            print("Invalid prediction:", prediction)
        mfccs = np.vstack([mfccs, prediction[0]])

    # print("Generated MFCCs:", mfccs)
    print("Generated mels shape:", mfccs.shape)
    #dump to json
    with open("generated_mels.json", "w") as fp:
        json.dump(mfccs.tolist(), fp, indent=4)
    #convert to audio
    audio = librosa.feature.inverse.mfcc_to_audio(mfccs.T ,n_fft=config["n_fft"], hop_length=config["hop_length"], sr=config["sample_rate"])
    sf.write("audio_from_mlp.wav", audio, config["sample_rate"], format='WAV', subtype='PCM_16')

def predict_mels():
    input_data = json.load(open("test_input_mels.json", "r"))["test_mel_1"]
    config = json.load(open("config.json", "r"))[MODEL_NAME]

    mels_to_generate = math.floor(config["sample_rate"]*config["duration"]/config["hop_length"])
    
    mels = np.array([input_data])
    for _ in range(mels_to_generate):
        prediction = predict(mels[-1])
        if np.any(np.isnan(prediction)) or np.any(np.isinf(prediction)):
            print("Invalid prediction:", prediction)
        mels = np.vstack([mels, prediction[0]])

    print("Generated Mels shape:", mels.shape)
    with open("generated.json", "w") as fp:
        json.dump(mels.tolist(), fp, indent=4)

def predict_mels_db():
    input_data = json.load(open("test_input_mels.json", "r"))["test_mel_2"]
    config = json.load(open("config.json", "r"))[MODEL_NAME]

    mels_to_generate = math.floor(config["sample_rate"]*config["duration"]/config["hop_length"])
    
    mels = np.array([input_data])
    for _ in range(mels_to_generate):
        prediction = predict(mels[-1])
        if np.any(np.isnan(prediction)) or np.any(np.isinf(prediction)):
            print("Invalid prediction:", prediction)
        mels = np.vstack([mels, prediction[0]])

    mels = librosa.db_to_power(mels)

    print("Generated Mels shape:", mels.shape)
    with open("generated_mel_db.json", "w") as fp:
        json.dump(mels.tolist(), fp, indent=4)
    

if __name__ == "__main__":
    # eval_mfcc_to_audio()
    predict_mels_db()