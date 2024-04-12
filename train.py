import numpy as np
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

DATASET_PATH = "data.json"

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    inputs = np.array(data["mfccs_input"])
    targets = np.array(data["mfccs_target"])

    print(f"inputs shape: {inputs.shape}")
    print(f"targets shape: {targets.shape}")

    return inputs, targets

def prepare_data(inputs, targets):
    inputs_new = inputs[:, :-1, :].reshape(-1, 13)
    targets_new = targets[:, 1:, :].reshape(-1, 13)
    return inputs_new, targets_new

if __name__ == "__main__":
    inputs, targets = load_data(DATASET_PATH)

    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, 
                                                                              targets, 
                                                                              test_size=0.3)
    
    inputs_train_new, targets_train_new = prepare_data(inputs_train, targets_train)
    inputs_test_new, targets_test_new = prepare_data(inputs_test, targets_test)
    
    print(inputs.shape[2])
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_shape=(13,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(13, activation="linear"),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss=tf.keras.losses.MeanSquaredError())
    
    model.summary()

    history = model.fit(inputs_train_new, targets_train_new,
                        validation_data=(inputs_test_new, targets_test_new),
                        epochs=50,
                        batch_size=32)
    
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="test loss")
    plt.show()

    model.save("model.h5")

   


   