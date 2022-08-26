from typing import Tuple, Union

import numpy as np
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
