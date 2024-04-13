import numpy as np
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

DATASET_PATH = "data.json"
CONFIG_PATH = "config.json"
MODEL_NAME = "first_prototype"

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    inputs = np.array(data["mfccs_input"])
    targets = np.array(data["mfccs_target"])

    print(f"inputs shape: {inputs.shape}")
    print(f"targets shape: {targets.shape}")

    return inputs, targets

def prepare_data(inputs, targets, test_size, n_mfcc):
    x = inputs[:, :-1, :].reshape(-1, n_mfcc)
    y = targets[:, 1:, :].reshape(-1, n_mfcc)
    return train_test_split(x, y, test_size=test_size)

if __name__ == "__main__":
    inputs, targets = load_data(DATASET_PATH)
    config = json.load(open(CONFIG_PATH, "r"))[MODEL_NAME]

    inputs_train, inputs_test, targets_train, targets_test = prepare_data(inputs, targets, test_size=config["test_size"], n_mfcc=config["n_mfcc"])
    
    print(inputs.shape[2])
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_shape=(config["n_mfcc"],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(13, activation="linear"),
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

   


   