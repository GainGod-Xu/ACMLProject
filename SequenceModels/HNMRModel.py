import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# import the necessary packages
def contrastive_loss(y, preds, margin=1):
    y = y.type(preds.dtype)
    squaredPreds = torch.square(preds)
    squaredMargin = torch.square(torch.maximum(margin - preds, 0))
    loss = torch.mean(y * squaredPreds + (1 - y) * squaredMargin)
    return loss


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, filters, window, num_channels):
        super(ConvolutionalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=filters, kernel_size=window, padding=window//2, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=filters, out_channels= filters//2, kernel_size=window, padding=window//2, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=filters//2, out_channels= filters//4, kernel_size=window, padding=window//2, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=filters//4, out_channels= filters//32, kernel_size=window, padding=window//2, stride=1),
            nn.ReLU()
            # Conv1D(filters/16, window, padding='same', strides=2, input_shape=input_shape, activation='relu'),
        )
        self.decoder = nn.Sequential(
            # Conv1DTranspose(filters/16, window, padding='same', strides=2, activation='relu'),
            nn.ConvTranspose1d(in_channels=filters//32, out_channels=filters//32, kernel_size=window, padding=window//2, stride=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=filters//32, out_channels=filters//4, kernel_size=window, padding=window//2, stride=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=filters//4, out_channels=filters//2, kernel_size=window, padding=window//2, stride=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=filters//2, out_channels=filters, kernel_size=window, padding=window//2, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=filters, out_channels=1, kernel_size=1, padding=0, stride=1)
        )

    def forward(self, inputs):
        inputs = inputs.float()
        out = self.encoder(inputs)
        out = out.squeeze(1) # remove channel dim to match with CMRP model
        # out = self.decoder(out)
        return out


class Autoencoder(nn.Module):
    def __init__(self, neurons, input_channel):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_channel, neurons[0]),
            nn.ReLU(),
            nn.Linear(neurons[0], neurons[1]),
            nn.ReLU(),
            nn.Linear(neurons[1], neurons[2]),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(neurons[2], neurons[1]),
            nn.ReLU(),
            nn.Linear(neurons[1], neurons[0]),
            nn.ReLU(),
            nn.Linear(neurons[0], input_channel),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.float()
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
