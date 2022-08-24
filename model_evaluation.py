"""
Author: your name
Date: 2021-03-16 13:11:17
LastEditTime: 2021-03-30 17:01:04
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /rnn_sc_wc/model_evaluation.py
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import torchvision.transforms as transforms
import seaborn as sns
import math

from models.model import ConvNet
from utils import (
    front_frame,
    input_frame,
    input_label,
    occlusion,
    RSA_predict,
)

pi = math.pi


def plot_filter(model, img="input/image.png"):
    model_children = list(model.children())

    # counter to keep count of the conv layers
    counter = 0

    model_weights = []  # we will save the conv layer weights in this list
    conv_layers = []  # we will save the conv layers in this list

    # append all the conv layers and their respective weights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for child in list(model_children[i]):
                if type(child) == nn.Conv2d:
                    counter += 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
    print(f"Total convolutional layers: {counter}")

    # take a look at the conv layers and the respective weights
    for weight, conv in zip(model_weights, conv_layers):
        # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
        print(f"CONV: {conv} ====> SHAPE: {weight.shape}")

    # visualize the first conv layer filters
    for layer_index in range(len(model_weights)):
        plt.figure(figsize=(15, 15))
        for i, filter in enumerate(model_weights[layer_index]):
            # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
            plt.subplot(8, 8, i + 1)
            plt.imshow(filter[0, :, :].detach(), cmap="gray")
            plt.axis("off")
            plt.title(i)
        plt.suptitle(f"Convolutional Layer Filter No.{layer_index}", fontsize=32)
        plt.savefig(f"output/filter{layer_index}.png")

        # define the transforms
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
        ]
    )

    img = np.array(img)
    # apply the transforms
    img = transform(img)
    print(img.size())
    # unsqueeze to add a batch dimension
    img = img.unsqueeze(0)
    print(img.size())

    # pass the image through all the layers
    results = [conv_layers[0](img)]
    for i in range(1, len(conv_layers)):
        # pass the result from the last layer to the next layer
        results.append(conv_layers[i](results[-1]))

    # make a copy of the `results`
    outputs = results

    # visualize 64 features from each layer
    # (although there are more feature maps in the upper layers)
    for num_layer in range(len(outputs)):
        plt.figure(figsize=(30, 30))
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        print(layer_viz.size())
        for i, filter in enumerate(layer_viz):
            if i == 64:  # we will visualize only 8x8 blocks from each layer
                break
            plt.subplot(8, 8, i + 1)
            plt.imshow(filter, cmap="gray")
            plt.axis("off")
        print(f"Saving layer {num_layer} feature maps...")
        plt.savefig(f"output/image_layer_{num_layer}.png")
        # plt.show()
        plt.close()

    pass


def generate_example_image():
    input_type = "SC"
    label_type = "SC"
    frames, start_poke_coordinate, target_poke_coordinate = front_frame(
        random_seed=20, frame_amount=1
    )
    image = input_frame(frames, input_type, start_poke_coordinate)
    label = input_label(
        start_poke_coordinate, target_poke_coordinate, label_type, "Cartesian"
    )

    img = image.astype(np.float32)
    img = np.expand_dims(img, 1)
    img = torch.from_numpy(img)
    plt.imshow(img[0][0])
    plt.show()
    lb = label.astype(np.float32)
    lb = torch.from_numpy(lb)

    plt.imsave("input/image.png", img[0][0].tolist(), dpi=100)
    # img = cv.imread(f"../input/{args['image']}")
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    pass


# def occlusion(
#     model, image="input/image.png", occ_size=50, occ_stride=50, occ_pixel=0.2
# ):

#     # get the width and height of the image
#     width, height = 100, 100
#     input_type = "SC"
#     label_type = "SC"
#     frames, start_poke_coordinate, target_poke_coordinate = front_frame(
#         random_seed=20, frame_amount=1
#     )
#     image = input_frame(frames, input_type, start_poke_coordinate)
#     image = np.expand_dims(image, 1)
#     image = torch.from_numpy(image)
#     # setting the output image width and height
#     output_height = int(np.ceil((height - occ_size) / occ_stride))
#     output_width = int(np.ceil((width - occ_size) / occ_stride))

#     # create a white image of sizes we defined
#     heatmap = torch.zeros((output_height, output_width))

#     # iterate all the pixels in each column
#     for h in range(0, height):
#         for w in range(0, width):

#             h_start = h * occ_stride
#             w_start = w * occ_stride
#             h_end = min(height, h_start + occ_size)
#             w_end = min(width, w_start + occ_size)

#             if (w_end) >= width or (h_end) >= height:
#                 continue

#             input_image = image.clone().detach()

#             # replacing all the pixel information in the image with occ_pixel(grey) in the specified location
#             input_image[:, :, w_start:w_end, h_start:h_end] = occ_pixel

#             # run inference on modified image
#             output = model(input_image.float())
#             output = nn.functional.softmax(output, dim=1)
#             prob = np.max(output.tolist())
#             # prob = np.max(output.tolist()[0])

#             # setting the heatmap location to probability value
#             heatmap[h, w] = prob
#
#    return heatmap


def rsa_visualization(rsa, label_type):

    if label_type == "Polar":
        save_name = "./figures/RSA_SC-SC_SC-WC_polar"
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_figheight(5)
        fig.set_figwidth(12)
        fig.suptitle("Comparison between SC-SC model and SC-WC model", fontsize=16)
        img = ax1.imshow(rsa[0])
        plt.colorbar(img, ax=ax1)
        ax1.set_title("Predict Theta value")
        ax1.set(xlabel="SC-SC", ylabel="SC-WC")
        ax1.set_xticks(range(0, len(rsa[0][0]), math.floor(len(rsa[0][0]) / 4)))
        ax1.set_xticklabels(["0", "pi/2", "pi", "3pi/2", "2pi"])
        ax1.set_yticks(range(0, len(rsa[0][0]), math.floor(len(rsa[0][0]) / 4)))
        ax1.set_yticklabels(["2pi", "3pi/2", "pi", "pi/2", "0"])
        img2 = ax2.imshow(rsa[1])
        plt.colorbar(img2, ax=ax2)
        ax2.set_title("Predict r value")
        ax2.set(xlabel="SC-SC", ylabel="SC-WC")
        ax2.set_xticks(range(0, len(rsa[0][0]), math.floor(len(rsa[0][0]) / 4)))
        ax2.set_xticklabels(["0", "pi/2", "pi", "3pi/2", "2pi"])
        ax2.set_yticks(range(0, len(rsa[0][0]), math.floor(len(rsa[0][0]) / 4)))
        ax2.set_yticklabels(["2pi", "3pi/2", "pi", "pi/2", "0"])
        plt.show()
    elif label_type == "Cartesian":
        save_name = "./figures/RSA_SC-SC_SC-WC_Carte"
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_figheight(5)
        fig.set_figwidth(12)
        fig.suptitle("Comparison between SC-SC model and SC-WC model", fontsize=16)
        img = ax1.imshow(rsa[0])
        plt.colorbar(img, ax=ax1)
        ax1.set_title("Predict X value")
        ax1.set(xlabel="SC-SC", ylabel="SC-WC")
        ax1.set_xticks(range(0, len(rsa[0][0]), math.floor(len(rsa[0][0]) / 4)))
        ax1.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
        ax1.set_yticks(range(0, len(rsa[0][0]), math.floor(len(rsa[0][0]) / 4)))
        ax1.set_yticklabels(["1", "0.75", "0.5", "0.25", "0"])
        img2 = ax2.imshow(rsa[1])
        plt.colorbar(img2, ax=ax2)
        ax2.set_title("Predict Y value")
        ax2.set(xlabel="SC-SC", ylabel="SC-WC")
        ax2.set_xticks(range(0, len(rsa[0][0]), math.floor(len(rsa[0][0]) / 4)))
        ax2.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
        ax2.set_yticks(range(0, len(rsa[0][0]), math.floor(len(rsa[0][0]) / 4)))
        ax2.set_yticklabels(["1", "0.75", "0.5", "0.25", "0"])
        plt.savefig(save_name + ".png", dpi=400)
        plt.show()

    pass


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generate_example_image()
    # ---------------------------------------------------------------------------- #
    #                                  load model                                  #
    # ---------------------------------------------------------------------------- #

    model = ConvNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    checkpoint = torch.load("model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]

    model2 = ConvNet()
    optimizer2 = torch.optim.Adam(model.parameters(), lr=0.001)
    checkpoint2 = torch.load("model2.pt")
    model2.load_state_dict(checkpoint2["model_state_dict"])
    optimizer2.load_state_dict(checkpoint2["optimizer_state_dict"])
    epoch2 = checkpoint2["epoch"]

    # ---------------------------------------------------------------------------- #
    #                               occlusion heatmap                              #
    # ---------------------------------------------------------------------------- #
    model = model.float()
    heatmap = occlusion(model, occ_size=10, occ_stride=5)
    print(heatmap.shape)
    print(heatmap)
    imgplot = sns.heatmap(heatmap, xticklabels=False, yticklabels=False)
    figure = imgplot.get_figure()
    figure.savefig("./figures/svm_conf.png", dpi=400)

    # ---------------------------------------------------------------------------- #
    #                                 RSA analysis                                 #
    # ---------------------------------------------------------------------------- #
    rsa = RSA_predict(model, model2)
    rsa_visualization(rsa, "Cartesian")
