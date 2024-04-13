import numpy as np
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

model = tf.keras.models.load_model('first_prototype.h5')

def predict(data):
    data = np.array(data)
    data = np.expand_dims(data, axis=0)
    prediction = model.predict(data)
    return prediction

if __name__ == "__main__":
    data = [
            20,
            -150.23976135253906,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
            ]
    mfccs = np.array([data])
    for _ in range(218):
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
    audio = librosa.feature.inverse.mfcc_to_audio(mfccs.T)
    sf.write("audio_from_mlp.wav", audio, 22500, format='WAV', subtype='PCM_16')

