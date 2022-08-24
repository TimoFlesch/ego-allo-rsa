"""
Author: your name
Date: 2021-03-17 11:27:00
LastEditTime: 2021-04-01 12:02:43
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /rnn_sc_wc/main.py
"""
import matplotlib.pyplot as plt

# import torchvision
# import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn as nn

from ego_allo_rnns.models.ffwd import ConvNet

# training data preperation
from ego_allo_rnns.utils.utils import (
    fit_transform,
    front_frame,
    input_frame,
    input_label,
)

if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------------------- #
    #                               train first model                              #
    # ---------------------------------------------------------------------------- #

    # Hyper parameters
    num_epochs = 2
    num_classes = 2
    batch_size = 32
    learning_rate = 0.001
    PATH = "model.pt"

    input_type = "SC"
    output_type = "SC"
    label_type = "Cartesian"
    title = "Input type: " + input_type + " Label type: " + output_type
    save_name = (
        "./figures/cnn_pytorch_input_"
        + input_type
        + "_label_"
        + output_type
        + "_Cartesian"
    )

    frames, start_poke_coordinate, target_poke_coordinate = front_frame(
        random_seed=20, frame_amount=5000
    )
    x_train = input_frame(frames, input_type, start_poke_coordinate)
    y_train = input_label(
        start_poke_coordinate, target_poke_coordinate, output_type, label_type
    )
    x_train = np.expand_dims(x_train, 1)

    frames, start_poke_coordinate, target_poke_coordinate = front_frame(
        random_seed=30, frame_amount=500
    )
    x_test = input_frame(frames, input_type, start_poke_coordinate)
    y_test = input_label(
        start_poke_coordinate, target_poke_coordinate, output_type, label_type
    )

    x_test = np.expand_dims(x_test, 1)

    y_train = fit_transform(y_train, label_type)
    y_test = fit_transform(y_test, label_type)

    x_train = torch.from_numpy(x_train).split(batch_size)
    y_train = torch.from_numpy(y_train).split(batch_size)

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    # initialize the model
    model = ConvNet(num_classes).to(device)
    model = model.float()
    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=learning_rate / 100
    )

    # Train the model
    total_step = len(x_train)
    loss_each_epochs = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(zip(x_train, y_train)):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images.float())
            loss = criterion(outputs, labels.float())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, total_step, loss.item()
                    )
                )
            if i == len(x_train) - 1:
                loss_each_epochs.append(loss.item())

    # visualize the training process
    plt.plot(range(num_epochs), loss_each_epochs)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Loss")
    plt.show()

    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

    # prediction
    predicted = model(x_test.float().to(device))
    predicted = predicted.detach().cpu().numpy()
    y_test = y_test.detach().numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(y_test[:, 0], predicted[:, 0], color="red")
    ax1.set_title("Original vs. Predicted:  X ")
    _ = ax1.plot([0, 1], [0, 1])
    ax2.scatter(y_test[:, 1], predicted[:, 1], color="red")
    ax2.set_title("Original vs. Predicted:  Y ")
    # plt.legend()
    _ = ax2.plot([0, 1], [0, 1])
    plt.suptitle("Original vs Predicted values: SC-SC")
    plt.savefig(save_name + "_x.png", dpi=400)
    plt.show()

    # ---------------------------------------------------------------------------- #
    #                              train second model                              #
    # ---------------------------------------------------------------------------- #

    # Hyper parameters
    num_epochs = 5
    num_classes = 2
    batch_size = 32
    learning_rate = 0.001

    input_type = "SC"
    output_type = "SC"
    label_type = "Cartesian"
    title = "Input type: " + input_type + " Label type: " + output_type
    save_name = (
        "./figures/cnn_pytorch_input_"
        + input_type
        + "_label_"
        + output_type
        + "_"
        + label_type
    )

    frames, start_poke_coordinate, target_poke_coordinate = front_frame(
        random_seed=20, frame_amount=5000
    )
    x_train = input_frame(frames, input_type, start_poke_coordinate)
    y_train = input_label(
        start_poke_coordinate, target_poke_coordinate, output_type, label_type
    )
    x_train = np.expand_dims(x_train, 1)

    frames, start_poke_coordinate, target_poke_coordinate = front_frame(
        random_seed=30, frame_amount=500
    )
    x_test = input_frame(frames, input_type, start_poke_coordinate)
    y_test = input_label(
        start_poke_coordinate, target_poke_coordinate, output_type, label_type
    )

    x_test = np.expand_dims(x_test, 1)

    y_train = fit_transform(y_train, label_type)
    y_test = fit_transform(y_test, label_type)
    print(y_train)

    x_train = torch.from_numpy(x_train).split(batch_size)
    y_train = torch.from_numpy(y_train).split(batch_size)

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    # initialize the model
    model2 = ConvNet(num_classes).to(device)
    model2 = model2.float()
    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model2.parameters(), lr=learning_rate, weight_decay=learning_rate / 100
    )

    # Train the model
    total_step = len(x_train)
    loss_each_epochs = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(zip(x_train, y_train)):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model2(images.float())
            loss = criterion(outputs, labels.float())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, total_step, loss.item()
                    )
                )
            if i == len(x_train) - 1:
                loss_each_epochs.append(loss.item())

    # prediction
    predicted = model2(x_test.float().to(device))
    predicted = predicted.detach().cpu().numpy()
    y_test = y_test.detach().numpy()
    fig2, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(y_test[:, 0], predicted[:, 0], color="red")
    ax1.set_title("Original vs. Predicted:  X ")
    _ = ax1.plot([0, 1], [0, 1])
    ax2.scatter(y_test[:, 1], predicted[:, 1], color="red")
    ax2.set_title("Original vs. Predicted:  Y ")
    # plt.legend()
    _ = ax2.plot([0, 1], [0, 1])
    plt.savefig(save_name + "_x.png", dpi=400)
    plt.show()

    # ---------------------------------------------------------------------------- #
    #                                 model saving                                 #
    # ---------------------------------------------------------------------------- #

    # Save the model checkpoint
    torch.save(model.state_dict(), "model.ckpt")

    # Save lar
    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "model.pt",
    )

    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model2.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "model2.pt",
    )
