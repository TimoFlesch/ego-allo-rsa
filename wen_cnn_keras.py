"""
Author: Xuan Wen
Date: 2021-02-28 03:18:29
LastEditTime: 2021-03-30 14:24:17
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /rnn_sc_wc/CNN_regression_combined.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# training model construction
from tensorflow import keras

# training data preperation
from ego_allo_rnns.utils.utils import (fit_transform, front_frame, input_frame,
                                       input_label)

coord_type = "Cartesian"
input_type = "WC"
label_type = "SC"
title = "Input type: " + input_type + " Label type: " + label_type
save_name = (
    "./figures/cnn_keras_input_"
    + input_type
    + "_label_"
    + label_type
    + "_"
    + coord_type
)

frames, start_poke_coordinate, target_poke_coordinate = front_frame(
    random_seed=20, frame_amount=5000
)
x_train = input_frame(frames, input_type, start_poke_coordinate)
y_train = input_label(
    start_poke_coordinate, target_poke_coordinate, label_type, coord_type
)
x_train = np.expand_dims(x_train, axis=3)

frames, start_poke_coordinate, target_poke_coordinate = front_frame(
    random_seed=30, frame_amount=500
)
x_test = input_frame(frames, input_type, start_poke_coordinate)
y_test = input_label(
    start_poke_coordinate, target_poke_coordinate, label_type, coord_type
)
x_test = np.expand_dims(x_test, axis=3)

y_train = fit_transform(y_train, coord_type)
y_test = fit_transform(y_test, coord_type)


def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    # define the model input
    inputs = keras.Input(shape=inputShape)
    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs
        # CONV => RELU => BN => POOL
        x = keras.layers.Conv2D(f, (3, 3), padding="same", activation="relu")(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(16)(x)
    if regress:
        x = keras.layers.Dense(2, activation="linear")(x)
    # construct the CNN
    model = keras.Model(inputs, x)
    # return the CNN
    return model


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Abs Error")
    plt.plot(hist["epoch"], hist["mae"], label="Train Error")
    plt.plot(hist["epoch"], hist["val_mae"], label="Val Error")
    plt.ylim([0, np.max([hist["val_mae"], hist["mae"]])])
    plt.title(title + "    Mean Abs Error")
    plt.legend()
    plt.savefig(save_name + "_mae.png")

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Square Error")
    plt.plot(hist["epoch"], hist["mse"], label="Train Error")
    plt.plot(hist["epoch"], hist["val_mse"], label="Val Error")
    plt.ylim([0, np.max([hist["val_mse"], hist["mse"]])])
    plt.title(title + "    Mean Square Error")
    plt.legend()
    plt.savefig(save_name + "_mse.png")
    plt.show()


EPOCHS = 20
model = create_cnn(100, 100, 1, regress=True)
opt = keras.optimizers.Adam(lr=1e-3, decay=1e-3 / 100)
# opt = keras.optimizers.RMSprop(0.001)
model.compile(loss="mean_absolute_error", optimizer=opt, metrics=["mae", "mse"])

print(model.summary())

# train the model

# patience value to check the epoch number
early_stop = keras.callbacks.EarlyStopping(monitor="loss", patience=2)


print("[INFO] training model...")
history = model.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_test, y_test),
    epochs=EPOCHS,
    batch_size=32,
    callbacks=[early_stop],
)

plot_history(history)


test_predictions = model.predict(x_test)
test_predictions = test_predictions[:, 0]
test_labels = y_test[:, 0]
plt.scatter(test_labels, test_predictions)
plt.xlabel("True Values X")
plt.ylabel("Predictions X")
plt.axis("equal")
plt.axis("square")
plt.title(title + "    X ")
plt.xlim([0, 1])
plt.ylim([0, 1])
_ = plt.plot([0, 1], [0, 1])
plt.savefig(save_name + "_x.png")
plt.show()


test_predictions = model.predict(x_test)
test_predictions = test_predictions[:, 1]
test_labels = y_test[:, 1]
plt.scatter(test_labels, test_predictions)
plt.xlabel("True Values Y")
plt.ylabel("Predictions Y")
plt.axis("equal")
plt.axis("square")
plt.title(title + "    Y ")
plt.xlim([0, 1])
plt.ylim([0, 1])
_ = plt.plot([0, 1], [0, 1])
plt.savefig(save_name + "_y.png")
plt.show()
