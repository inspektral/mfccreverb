import numpy as np
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

DATASET_PATH = "data_mels_db.json"
CONFIG_PATH = "config.json"
MODEL_NAME = "mel_db_hifi"

def load_data_mfccs(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    mfccs = np.array(data["mfccs"])

    return mfccs

def load_data_mels(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    mels = np.array(data["mels"])

    return mels

def prepare_data(data, test_size, col_size):
    x = data[:, :-1, :].reshape(-1, col_size)
    y = data[:, 1:, :].reshape(-1, col_size)
    return train_test_split(x, y, test_size=test_size)

if __name__ == "__main__":
    mels = load_data_mels(DATASET_PATH)
    config = json.load(open(CONFIG_PATH, "r"))[MODEL_NAME]

    inputs_train, inputs_test, targets_train, targets_test = prepare_data(mels, test_size=config["test_size"], col_size=config["n_mels"])
        
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_shape=(config["n_mels"],), activation="relu",),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(config["n_mels"], activation="linear")
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
                    loss=tf.keras.losses.MeanSquaredError())
    
    model.summary()

    history = model.fit(inputs_train, targets_train,
                        validation_data=(inputs_test, targets_test),
                        epochs=config["epochs"],
                        batch_size=config["batch_size"])
    
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="test loss")
    plt.show()

    model.save(MODEL_NAME+".h5")

   


   