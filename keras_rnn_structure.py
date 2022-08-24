"""
Author: Xuan Wen
Date: 2021-02-22 16:59:39
LastEditTime: 2021-03-10 15:01:00
Description: In User Settings Edit
FilePath: /rnn_sc_wc/keras_rnn_structure.py
"""
from tensorflow import keras
from utils import input_frame, front_frame, input_label


frames, start_poke_coordinate, target_poke_coordinate = front_frame(
    random_seed=20, frame_amount=10000
)
x_train = input_frame(frames, "WC", start_poke_coordinate)
y_train = input_label(start_poke_coordinate, target_poke_coordinate, "WC", "Cartesian")

frames, start_poke_coordinate, target_poke_coordinate = front_frame(
    random_seed=30, frame_amount=5000
)
x_test = input_frame(frames, "WC", start_poke_coordinate)
y_test = input_label(start_poke_coordinate, target_poke_coordinate, "WC", "Cartesian")


def create_lstm(width, height, depth, time, filters=(16, 32, 64)):
    model = keras.Sequential(
        [
            keras.Input(
                shape=(None, width, height, depth)
            ),  # Variable-length sequence of w*h*d frames
            keras.layers.ConvLSTM2D(
                filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
            ),
            keras.layers.BatchNormalization(),
            keras.layers.ConvLSTM2D(
                filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
            ),
            keras.layers.BatchNormalization(),
            keras.layers.ConvLSTM2D(
                filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
            ),
            keras.layers.BatchNormalization(),
            keras.layers.ConvLSTM2D(
                filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Conv3D(
                filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
            ),
        ]
    )

    return model
