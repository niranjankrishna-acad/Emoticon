import torch

import torch.nn as nn


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        self.cnn_block_1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Dropout(p = 0.25)

        )

        self.cnn_block_2 =  nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Dropout(p = 0.25)

        )

        self.linear_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = 2048, out_features = 1024),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(in_features = 1024, out_features = 7),
            nn.Softmax(),
        )

    def forward(self, x):

        x = self.cnn_block_1(x)
        x = self.cnn_block_2(x)
        x = self.linear_block(x)

        return x
