from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import TensorDataset

from ego_allo_rnns.utils.utils import front_frame, input_frame, input_label


def make_datasets(
    size_ds: Union[None, int] = None,
    input_type: str = "WC",
    output_type: str = "WC",
    label_type: str = "Cartesian",
    n_train: int = 5000,
    n_test: int = 200,
    random_seed: int = 20,
    n_frames: int = 11,
    target_frame: Tuple[int, int] = (4, 4),
) -> Tuple[TensorDataset, TensorDataset]:
    """generates training and test datasets

    Args:
        size_ds (Union[None, int], optional): downsampled size of image in pixels. Defaults to None.
        input_type (string): WC or SC. Defaults to WC.
        output_type (string): WC or SC. Defaults to WC.
        label_type (string): Cartesian or Polar. Defaults to Cartesian.
        n_train (int, optional): number of training samples. Defaults to 5000.
        n_test (int, optional): number of test samples. Defaults to 200.
        random_seed (int, optional): seed for rng. Defaults to 200.
        n_frames (int, optional): number of frames per trial. Defaults to 11.

        target_frame (tuple, optional): Frame that shows target. Defaults to U(2,4).

    Returns:
        Tuple[TensorDataset, TensorDataset]: training and test sets with labels
    """
    size_ds = size_ds or 100
    resize = T.Resize(
        size_ds, interpolation=T.InterpolationMode.NEAREST, antialias=False
    )

    # generate training dataset
    frames, start_poke_coordinate, target_poke_coordinate = front_frame(
        random_seed=random_seed, frame_amount=n_train, show_target=False
    )
    x_train_notarget = input_frame(
        frames, input_type=input_type, start_poke=start_poke_coordinate
    )
    x_train_notarget = resize(torch.tensor(x_train_notarget)).numpy()
    frames, start_poke_coordinate, target_poke_coordinate = front_frame(
        random_seed=random_seed, frame_amount=n_train, show_target=True
    )
    x_train_withtarget = input_frame(
        frames, input_type=input_type, start_poke=start_poke_coordinate
    )
    x_train_withtarget = resize(torch.tensor(x_train_withtarget)).numpy()

    x_train = np.repeat(x_train_notarget[:, np.newaxis, :, :], n_frames, axis=1)
    targets = np.random.uniform(
        target_frame[0], target_frame[1] + 1, size=n_train
    ).astype(int)
    for i, t in enumerate(targets):
        x_train[i, t, :, :] = x_train_withtarget[i, :, :]
    dims = x_train.shape
    x_train = torch.tensor(
        x_train.reshape((*dims[:2], dims[-1] ** 2)), dtype=torch.float
    )
    y_train = torch.tensor(
        input_label(
            start_poke_coordinate, target_poke_coordinate, output_type, label_type
        ),
        dtype=torch.float,
    )

    # generate test set
    frames, start_poke_coordinate, target_poke_coordinate = front_frame(
        random_seed=random_seed * 20, frame_amount=n_test, show_target=False
    )
    x_test_notarget = input_frame(
        frames, input_type=input_type, start_poke=start_poke_coordinate
    )
    x_test_notarget = resize(torch.tensor(x_test_notarget)).numpy()
    frames, start_poke_coordinate, target_poke_coordinate = front_frame(
        random_seed=random_seed * 20, frame_amount=n_test, show_target=True
    )
    x_test_withtarget = input_frame(
        frames, input_type=input_type, start_poke=start_poke_coordinate
    )
    x_test_withtarget = resize(torch.tensor(x_test_withtarget)).numpy()

    x_test = np.repeat(x_test_notarget[:, np.newaxis, :, :], n_frames, axis=1)
    targets = np.random.uniform(
        target_frame[0], target_frame[1] + 1, size=n_test
    ).astype(int)
    for i, t in enumerate(targets):
        x_test[i, t, :, :] = x_test_withtarget[i, :, :]

    dims = x_test.shape
    x_test = torch.tensor(x_test.reshape((*dims[:2], dims[-1] ** 2)), dtype=torch.float)
    y_test = torch.tensor(
        input_label(
            start_poke_coordinate, target_poke_coordinate, output_type, label_type
        ),
        dtype=torch.float,
    )

    return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)


def make_testset_from_conditions(
    df: Union[pd.DataFrame, None] = None,
    random_seed: int = 123,
    input_type: str = "WC",
    output_type: str = "WC",
    label_type: str = "Cartesian",
    n_frames: int = 11,
    target_frame: int = (4, 4),
    size_ds: int = 40,
) -> Tuple[pd.DataFrame, TensorDataset]:
    """creates test dataset based on user-specified condition table

    Args:
        df (Union[pd.DataFrame, None], optional): dataframe with conditions. Defaults to None.
        random_seed (int, optional): seed for noise pokes. Defaults to 123.
        input_type (string): WC or SC. Defaults to WC.
        output_type (string): WC or SC. Defaults to WC.
        label_type (string): Cartesian or Polar. Defaults to Cartesian.
        n_frames (int, optional): number of frames per trial. Defaults to 11.
        target_frame (tuple, optional): Frame that shows target. Defaults to U(4,4).

    Returns:
        Tuple[pd.DataFrame, TensorDataset]: condition table and dataset with x and y
    """

    resize = T.Resize(
        size_ds, interpolation=T.InterpolationMode.NEAREST, antialias=False
    )

    df = df or make_conditiontable()

    start_coord = df[["start_loc_y", "start_loc_x"]].to_numpy().T
    target_coord = df[["target_loc_y", "target_loc_x"]].to_numpy().T
    n_trials = start_coord.shape[1]

    frames, start_poke_coordinate, target_poke_coordinate = front_frame(
        random_seed=random_seed,
        frame_amount=n_trials,
        start_coordinate=start_coord,
        target_coordinate=target_coord,
        show_target=False,
    )
    x_test_notarget = input_frame(
        frames, input_type=input_type, start_poke=start_poke_coordinate
    )
    x_test_notarget = resize(torch.tensor(x_test_notarget)).numpy()

    frames, start_poke_coordinate, target_poke_coordinate = front_frame(
        random_seed=random_seed,
        frame_amount=n_trials,
        start_coordinate=start_coord,
        target_coordinate=target_coord,
        show_target=True,
    )
    x_test_withtarget = input_frame(
        frames, input_type=input_type, start_poke=start_poke_coordinate
    )
    x_test_withtarget = resize(torch.tensor(x_test_withtarget)).numpy()

    x_test = np.repeat(x_test_notarget[:, np.newaxis, :, :], n_frames, axis=1)

    targets = np.random.uniform(
        target_frame[0], target_frame[1] + 1, size=n_trials
    ).astype(int)
    for i, t in enumerate(targets):
        # slot in frame with target
        x_test[i, t, :, :] = x_test_withtarget[i, :, :]

    dims = x_test.shape
    x_test = torch.tensor(x_test.reshape((*dims[:2], dims[-1] ** 2)), dtype=torch.float)
    y_test = torch.tensor(
        input_label(
            start_poke_coordinate, target_poke_coordinate, output_type, label_type
        ),
        dtype=torch.float,
    )
    data_test = TensorDataset(x_test, y_test)

    return df, data_test


def make_conditiontable() -> pd.DataFrame:
    """creates condition table that covers 24 combinations of target directions and locations

    Returns:
        pd.DataFrame: condition table
    """

    x_locs = np.array([16, 32, 8, 24, 40, 16, 32]) + 1
    y_locs = np.array([16, 16, 24, 24, 24, 32, 32]) + 1
    trial_types = np.array(
        [
            [(0, 2), (1, 3), (3, 5), (4, 6)],
            [(0, 3), (1, 4), (2, 5), (3, 6)],
            [(1, 0), (3, 2), (4, 3), (6, 5)],
            [(0, 1), (2, 3), (3, 4), (5, 6)],
            [(3, 0), (4, 1), (5, 2), (6, 3)],
            [(2, 0), (3, 1), (5, 3), (6, 4)],
        ]
    )
    direction_colors = [
        "royalblue",
        "darkred",
        "dodgerblue",
        "indianred",
        "lightblue",
        "pink",
    ]
    location_colors = [
        "palegreen",
        "lightsalmon",
        "mediumseagreen",
        "olive",
        "goldenrod",
        "green",
        "sienna",
    ]
    header = [
        "start_loc_id",
        "target_loc_id",
        "target_direction_id",
        "start_loc_x",
        "start_loc_y",
        "target_loc_x",
        "target_loc_y",
        "target_loc_color",
        "target_direction_color",
    ]
    rows = []
    for i, dc in enumerate(direction_colors):  # for each motion direction
        for j in range(4):  # for each trial of that direction
            # add pairs of start and goal locations
            idces = trial_types[i, j, :]
            rows.append(
                [
                    idces[0],
                    idces[1],
                    i,
                    x_locs[idces[0]],
                    y_locs[idces[0]],
                    x_locs[idces[1]],
                    y_locs[idces[1]],
                    location_colors[idces[1]],
                    dc,
                ]
            )
    df = pd.DataFrame(np.asarray(rows), columns=header)
    cs = header[:-2]
    cs
    for c in cs:
        df[c] = df[c].astype("int32")

    return df
