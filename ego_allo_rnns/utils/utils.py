import math
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from moviepy.editor import ImageSequenceClip
from numpy import random
from tensorflow.python.keras import backend as K


def random_poke_generator(num_element=5, poke_size=3):
    # fix random seed to fix output numbers
    samples = (50 - poke_size - 1) * random.sample(size=2 * num_element) + 1
    samples = np.reshape(samples.astype(int), (2, num_element))
    return samples


def front_frame(
    random_seed: int = 20, frame_amount: int = 20, show_target: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """generates frames with stimulus information
        Each stimulus consists of an image with start and target ports
        and a few faint distractor ports
        The intensity indicates whether the port is a distractor (0.2), start (0.7) or target (1.0)

    Args:
        random_seed (int, optional): seed for rng. Defaults to 20.
        frame_amount (int, optional): number of frames to generate. Defaults to 20.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: frames, coords of start and target ports
    """
    random.seed(random_seed)
    front_frame_size = 50
    poke_size = 5
    frames = np.zeros((frame_amount, front_frame_size, front_frame_size))
    frames[:][:][:] = 0.1
    start_poke_coordinate = random_poke_generator(frame_amount, poke_size=poke_size)
    target_poke_coordinate = random_poke_generator(frame_amount, poke_size=poke_size)

    for num in range(0, frame_amount):
        empty_poke_coordinate = random_poke_generator(poke_size=poke_size)
        # add noise pokes
        for index in range(0, len(empty_poke_coordinate[0])):
            for poke_row in range(poke_size):
                for poke_column in range(poke_size):
                    frames[num][empty_poke_coordinate[0][index] + poke_row][
                        empty_poke_coordinate[1][index] + poke_column
                    ] = 0.2
        # add start poke
        for poke_row in range(poke_size):
            for poke_column in range(poke_size):
                frames[num][start_poke_coordinate[0][num] + poke_row][
                    start_poke_coordinate[1][num] + poke_column
                ] = 0.7

        # add target poke
        if show_target:
            for poke_row in range(poke_size):
                for poke_column in range(poke_size):
                    frames[num][target_poke_coordinate[0][num] + poke_row][
                        target_poke_coordinate[1][num] + poke_column
                    ] = 1

    return frames, start_poke_coordinate, target_poke_coordinate


def input_frame(front_frame, output_type, start_poke=None):
    """generate number of frames can be used as input
    attach the front frame to the background frame, which centered at the start poke (SC) or the actual center (WC)

    Args:
        front_frame (ndarray): frame_amount * 50 * 50
        output_type (string): WC or SC
        start_poke (ndarray): frame_amount * 2, set to None as default

    Returns:
        combined_frame: frame_amount * 100 * 100
    """
    background_size = 100
    conbined_frame = np.zeros((len(front_frame), background_size, background_size))
    if output_type == "WC":
        for index in range(0, len(front_frame)):
            conbined_frame[index, 24 : 24 + 50, 24 : 24 + 50] = front_frame[index]
    elif output_type == "SC" and start_poke.all() is not None:
        for index in range(0, len(front_frame)):
            # conbined_frame[a, b, c]
            conbined_frame[
                index,
                50 - start_poke[0][index] : 100 - start_poke[0][index],
                50 - start_poke[1][index] : 100 - start_poke[1][index],
            ] = front_frame[index]
    else:
        return 0
    return conbined_frame


def input_label(start_poke, target_poke, output_type, label_type):
    """generate training label by its start poke and target poke

    Args:
        start_poke ([ndarray]): frame amount * 2
        target_poke (ndarray): frame amount * 2
        output_type (string): SC or WC
        label_type (string): cartesian or polar

    Returns:
        ndarray: frame amount * 2 --> x,y for cartesian and Theta, r for polar
    """
    label = np.zeros((len(start_poke[0]), 2))
    for index in range(0, len(start_poke[0])):
        if output_type == "WC":

            if label_type == "Cartesian":
                label[index] = world_center_distance_cartesian(
                    target=[target_poke[0][index], target_poke[1][index]]
                )
            elif label_type == "Polar":
                label[index] = world_center_distance_polar(
                    target=[target_poke[0][index], target_poke[1][index]]
                )
            else:
                return 0

        elif output_type == "SC":

            if label_type == "Cartesian":
                label[index] = self_center_distance_cartesian(
                    start=[start_poke[0][index], start_poke[1][index]],
                    target=[target_poke[0][index], target_poke[1][index]],
                )
            elif label_type == "Polar":
                label[index] = self_center_distance_polar(
                    start=[start_poke[0][index], start_poke[1][index]],
                    target=[target_poke[0][index], target_poke[1][index]],
                )
            else:
                return 0

        else:
            return 0

    return label


def distance_difference(x, y):
    """
    Calculate minimum difference between two points
    """
    return abs(x - y)


def fit_transform(data, label_type=None):
    data = data.astype(float)
    if label_type == "Polar":
        scaler = [2 * math.pi, 50]
    elif label_type == "Cartesian":
        scaler = [
            abs(np.max(data[:, 0]) - np.min(data[:, 0])),
            np.max(data[:, 1]) - np.min(data[:, 1]),
        ]
    else:
        return None
    for i in np.arange(len(data[0])):
        s = scaler[i]
        data[:, i] = (data[:, i] - np.min(data[:, i])) * 1 / s
    return data


def distance_error(y_true, y_pred):
    """
    Calculate the mean diference between the true points
    and the predicted points. Each point is represented
    as a binary vector.
    """
    diff = distance_difference(K.argmax(y_true), K.argmax(y_pred))
    return K.mean(K.cast(K.abs(diff), K.floatx()))


def distance_error_regression(y_true, y_pred):
    return K.mean(distance_difference(y_true, y_pred))


def label_unification(type, number):
    if type == "Cartesian":
        multiplier = math.floor(number / 5)
        number = 5 * multiplier
    if type == "Polar":
        multiplier = math.floor(number / 0.2)
        number = 0.2 * multiplier

    return number


def float_points_in_circum(r, n=150):
    return [
        (math.cos(2 * math.pi / n * x) * r, math.sin(2 * math.pi / n * x) * r)
        for x in range(0, n + 1)
    ]


def get_angle(coordinate):
    r = 24
    return math.atan2(coordinate[1] - r, coordinate[0] - r)


def circum(r=10):
    unsorted = np.unique(
        np.asarray(
            [
                np.asarray([math.floor(float(i)) for i in x]) + r
                for x in float_points_in_circum(r)
            ]
        ),
        axis=0,
    )
    sort = np.asarray(sorted(unsorted, key=get_angle))
    return sort


def ordered_xy_poke_generator(fix="y"):
    start_poke = np.zeros((2, 50))
    start_poke[:][:] = 24

    start_poke[:][:] = 24
    if fix == "y":
        poke = np.zeros((2, 50))
        poke[0] = np.arange(0, 50, 1)
        poke[1][:] = 25
        return start_poke.astype(int), poke.T.astype(int)
    elif fix == "x":
        poke = np.zeros((2, 50))
        poke[1] = np.arange(0, 50, 1)
        poke[0][:] = 25
        return start_poke.astype(int), poke.T.astype(int)
    else:
        return None


def ordered_circle_poke_generator():
    target_poke = circum(r=24)
    start_poke = np.zeros((2, len(target_poke)))
    start_poke[:][:] = 24

    return start_poke.astype(int), target_poke.T.astype(int)


def ordered_front_frame():
    front_frame_size = 50
    poke_size = 2
    start_poke_coordinate, target_poke_coordinate = ordered_circle_poke_generator()
    frame_amount = len(target_poke_coordinate[0])
    print(frame_amount)
    frames = np.zeros((frame_amount, front_frame_size, front_frame_size))
    frames[:][:][:] = 0.1
    print(frames.shape)
    for num in range(0, frame_amount):
        # add start poke
        for poke_row in range(poke_size):
            for poke_column in range(poke_size):
                frames[num][start_poke_coordinate[0][num] + poke_row][
                    start_poke_coordinate[1][num] + poke_column
                ] = 0.7
        # add target poke
        for poke_row in range(poke_size):
            for poke_column in range(poke_size):
                frames[num][target_poke_coordinate[0][num] + poke_row][
                    target_poke_coordinate[1][num] + poke_column
                ] = 1
        frames[num] = np.rot90(frames[num], k=3)

    return frames, start_poke_coordinate, target_poke_coordinate


# start/target = [x, y] in cartesian, return value vector:[m, n], polar: [Theta, distance]
# ! Convert the coordinate from [0,0] at the top-left to [0,0] at the center (only needed in the world-center)
def coordinate_shift(x, y, frame_size=50):
    center = math.ceil(frame_size / 2)
    return [x - center, y - center]


def world_center_distance_cartesian(target):
    vector = coordinate_shift(*target)
    return vector


def world_center_distance_polar(target):

    vector = coordinate_shift(*target)
    # if vector[0] == 0 and vector[1] != 0:
    #     return [math.pi/2 * vector[1]/abs(vector[1]), math.sqrt(math.pow(vector[0], 2) + math.pow(vector[1], 2))]
    # elif vector[0] == 0 and vector[1] == 0:
    #     return [0, 0]
    # # elif vector[0] is -0.0:
    # #     print('is 0.0')
    # #     return [-math.pi, math.sqrt(math.pow(vector[0], 2) + math.pow(vector[1], 2))]
    # # elif vector[0] is 0.0:
    # #     print('is 0.0')
    # #     return [math.pi, math.sqrt(math.pow(vector[0], 2) + math.pow(vector[1], 2))]
    # else:
    return [
        math.atan2(vector[1], vector[0]),
        math.sqrt(math.pow(vector[0], 2) + math.pow(vector[1], 2)),
    ]


def self_center_distance_cartesian(start, target):
    return [target[0] - start[0], target[1] - start[1]]
    # return [start[0] - target[0], start[1] - target[1]]


def self_center_distance_polar(start, target):

    x = target[0] - start[0]
    y = target[1] - start[1]
    # if x == 0 and y != 0:
    #     return [math.pi/2 * y/abs(y), math.sqrt(math.pow(x, 2) + math.pow(y, 2))]
    # elif x == 0 and y == 0:
    #     return [0, 0]
    # # if x is -0.0:
    #     return [-math.pi, math.sqrt(math.pow(x, 2) + math.pow(y, 2))]
    # elif x is 0.0:
    #     return [math.pi, math.sqrt(math.pow(x, 2) + math.pow(y, 2))]
    # else:
    return [math.atan2(y, x), math.sqrt(math.pow(x, 2) + math.pow(y, 2))]


def plot_filters_single_channel_big(t):

    # setting the rows and columns
    nrows = t.shape[0] * t.shape[2]
    ncols = t.shape[1] * t.shape[3]

    npimg = np.array(t.numpy(), np.float32)
    npimg = npimg.transpose((0, 2, 1, 3))
    npimg = npimg.ravel().reshape(nrows, ncols)

    npimg = npimg.T

    fig, ax = plt.subplots(figsize=(ncols / 10, nrows / 200))
    imgplot = sns.heatmap(  # noqa F841
        npimg, xticklabels=False, yticklabels=False, cmap="gray", ax=ax, cbar=False
    )
    pass


def plot_filters_single_channel(t):

    # kernels depth * number of kernels
    nplots = t.shape[0] * t.shape[1]
    ncols = 12

    nrows = 1 + nplots // ncols
    # convert tensor to numpy image
    npimg = np.array(t.numpy(), np.float32)

    count = 0
    fig = plt.figure(figsize=(ncols, nrows))

    # looping through all the kernels in each channel
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            count += 1
            ax1 = fig.add_subplot(nrows, ncols, count)
            npimg = np.array(t[i, j].numpy(), np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            ax1.imshow(npimg)
            ax1.set_title(str(i) + "," + str(j))
            ax1.axis("off")
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

    plt.tight_layout()
    plt.show()
    pass


def plot_filters_multi_channel(t):

    # get the number of kernals
    num_kernels = t.shape[0]

    # define number of columns for subplots
    num_cols = 12
    # rows = num of kernels
    num_rows = num_kernels

    # set the figure size
    fig = plt.figure(figsize=(num_cols, num_rows))

    # looping through all the kernels
    for i in range(t.shape[0]):
        ax1 = fig.add_subplot(num_rows, num_cols, i + 1)

        # for each kernel, we convert the tensor to numpy
        npimg = np.array(t[i].numpy(), np.float32)
        # standardize the numpy image
        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        ax1.imshow(npimg)
        ax1.axis("off")
        ax1.set_title(str(i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.savefig("myimage.png", dpi=100)
    plt.tight_layout()
    plt.show()
    pass


def plot_weights(model, layer_num, single_channel=True, collated=False):

    # extracting the model features at the particular layer number
    layer = model.features[layer_num]

    # checking whether the layer is convolution layer or not
    if isinstance(layer, nn.Conv2d):
        # getting the weight tensor data
        weight_tensor = model.features[layer_num].weight.data

        if single_channel:
            if collated:
                plot_filters_single_channel_big(weight_tensor)
            else:
                plot_filters_single_channel(weight_tensor)

        else:
            if weight_tensor.shape[1] == 3:
                plot_filters_multi_channel(weight_tensor)
            else:
                print(
                    "Can only plot weights with three channels with single channel = False"
                )
    else:
        print("Can only visualize layers which are convolutional")


# custom function to conduct occlusion experiments


def occlusion(model, image, label, occ_size=50, occ_stride=50, occ_pixel=0.5):

    # get the width and height of the image
    width, height = image.shape[-2], image.shape[-1]

    # setting the output image width and height
    output_height = int(np.ceil((height - occ_size) / occ_stride))
    output_width = int(np.ceil((width - occ_size) / occ_stride))

    # create a white image of sizes we defined
    heatmap = torch.zeros((output_height, output_width))

    # iterate all the pixels in each column
    for h in range(0, height):
        for w in range(0, width):

            h_start = h * occ_stride
            w_start = w * occ_stride
            h_end = min(height, h_start + occ_size)
            w_end = min(width, w_start + occ_size)

            if (w_end) >= width or (h_end) >= height:
                continue

            input_image = image.clone().detach()

            # replacing all the pixel information in the image with occ_pixel(grey) in the specified location
            input_image[:, :, w_start:w_end, h_start:h_end] = occ_pixel

            # run inference on modified image
            output = model(input_image)
            output = nn.functional.softmax(output, dim=1)
            prob = output.tolist()[0][label]

            # setting the heatmap location to probability value
            heatmap[h, w] = prob

    return heatmap


def RSA_random_input_generate():
    input_type = "SC"
    label_type = "SC"
    frames, start_poke_coordinate, target_poke_coordinate = front_frame(
        random_seed=88, frame_amount=1000
    )
    input = input_frame(frames, input_type, start_poke_coordinate)
    label = input_label(
        start_poke_coordinate, target_poke_coordinate, label_type, "Cartesian"
    )

    # add index to the original label
    label_with_index = [(idx, item) for idx, item in enumerate(label)]
    # sort by the first element
    sort_by_x = sorted(label_with_index, key=lambda tup: tup[1][0])
    sort_by_y = sorted(label_with_index, key=lambda tup: tup[1][1])
    print(sort_by_x)

    # Unique Kth index tuples
    # Using map() + next() + lambda
    unique_x = [
        *map(
            lambda ele: next(tup for tup in sort_by_x if tup[1][0] == ele),
            {tup[1][0] for tup in sort_by_x},
        )
    ]

    unique_y = [
        *map(
            lambda ele: next(tup for tup in sort_by_y if tup[1][1] == ele),
            {tup[1][1] for tup in sort_by_y},
        )
    ]
    # get the index
    order_x = [x[0] for x in unique_x]
    order_y = [x[0] for x in unique_y]

    # use this order to select frame
    RSA_x = [input[i] for i in order_x]
    RSA_y = [input[i] for i in order_y]
    return RSA_x, RSA_y


def RSA_order_input_generate():
    input_type = "WC"
    label_type = "WC"
    frames, start_poke_coordinate, target_poke_coordinate = ordered_front_frame()
    input = input_frame(frames, input_type, start_poke_coordinate)
    label = input_label(
        start_poke_coordinate, target_poke_coordinate, label_type, "Polar"
    )
    return input, label


def RSA_input_generate():
    pass


def RSA_predict(model_1, model_2):
    num_class = 2
    input, label = RSA_order_input_generate()
    input = np.expand_dims(input, 1)
    input = torch.from_numpy(input)
    RSA_matrix_x = np.zeros((num_class, len(input), len(input)))  # build matrix

    predict_collect_1 = model_1(input.float())
    predict_collect_2 = model_2(input.float())
    predict_collect_1 = predict_collect_1.detach().numpy()
    predict_collect_2 = predict_collect_2.detach().numpy()
    # save_predict_value = np.zeros

    for i1, elem1 in enumerate(predict_collect_1):
        for i2, elem2 in enumerate(predict_collect_2):
            RSA_matrix_x[0][len(input) - 1 - i1][i2] = abs(elem1[0] - elem2[0])
            RSA_matrix_x[1][len(input) - 1 - i1][i2] = abs(elem1[1] - elem2[1])

    return RSA_matrix_x


def example_gif_generate():
    input_type = "WC"
    label_type = "WC"  # noqa F841
    frames, start_poke_coordinate, target_poke_coordinate = ordered_front_frame()
    frames = input_frame(frames, input_type, start_poke_coordinate)
    frames = abs(1 - frames) * 1000
    new_frames = np.zeros((len(frames), 500, 500))
    for i in range(len(frames)):
        new_frames[i] = cv2.resize(
            frames[i], dsize=(500, 500), interpolation=cv2.INTER_NEAREST
        )
    new_frames = np.expand_dims(new_frames, 3)
    clip = ImageSequenceClip(list(new_frames), fps=20)
    clip.write_gif("/Users/wen/repos/rnn_sc_wc/output/test_example.gif", fps=20)
    pass


if __name__ == "__main__":

    """
    # * show example input picture
    frames, start_poke_coordinate, target_poke_coordinate = front_frame(frame_amount=1)
    final_inputs = input_frame(frames, "SC", start_poke_coordinate)
    y_train = input_label(start_poke_coordinate, target_poke_coordinate, "SC", "Cartesian")
    print(y_train)
    print(start_poke_coordinate)
    print(target_poke_coordinate)
    for index in range(0, len(frames)):
        plt.imshow(final_inputs[index])
        plt.show()
    """
    frames, start_poke_coordinate, target_poke_coordinate = ordered_front_frame()
    for index in range(0, len(frames)):
        plt.imshow(frames[index])
        plt.show()
