"""
This is the module for the reward model.
The reward model, or discriminator, is composed of a visual module (CNN) for 
interpreting the image data, and of a temporal module (LSTM) for interpreting
the textual data.
"""
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

class FiLM_LSTM(nn.Module):
    """
    Implements the conditioning part of the FiLM layer.
    This LSTM consumes the instruction and returns it as a set of 
    conditionning parameters (multiplyer and biases) to be applyed to the 
    convolutional layers of the reward and policy models.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)  # just one layer for now
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # maybe provide initial hidden state and initial context
        output, (hn, cn) = self.lstm(x)
        out = F.relu(hn)
        out = self.linear(out)
        # another non-linearity ?
        # maybe reshape here for use by the conv layers
        return out

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 
                              out_channels,
                              kernel_size,
                              padding=padding)

    def forward(self, x, gamma, beta, residual=None):
        """
        Implements the FiLM conditioning.
        """
        x = self.conv(x)
        # batch norm ?
        x = (1 + gamma) * x + beta
        if residual:
            x = x + residual
        x = F.relu(x)
        return x

class Discriminator(nn.Module):

    def __init__(self, image_shape, input_size, output_size, hidden_size=100):
        super().__init__()
        self.image_shape = image_shape

        self.conv1 = ConvBlock(3, 16, 8, padding=1)
        self.conv2 = ConvBlock(16, 32, 3, padding=1)
        self.conv3 = ConvBlock(32, 64, 3)
        self.conv4 = ConvBlock(64, 64, 3)
        self.conv5 = ConvBlock(64, 64, 3)
        self.maxpool = nn.MaxPool2d(2, 2, padding=1)

        self.linear1 = nn.Linear(16384, 100)  # a bit much ?
        self.linear2 = nn.Linear(100, 1)

        self.film_lstm = FiLM_LSTM(input_size, hidden_size, output_size)

    def forward(self, x, s):
        """
        x: input image;
        s: input sentence sequence (after coding, embedding/whatever)
        """
        gamma, beta = self.film_lstm(s)  # shape this

        x = F.relu(self.conv1(x, gamma[0], beta[0]))
        x = F.relu(self.conv2(x, gamma[1], beta[1]))
        x3 = F.relu(self.conv3(x, gamma[2], beta[2]))
        x = F.relu(self.conv4(x, gamma[3], beta[3]))
        x = F.relu(self.conv5(x, gamma[4], beta[4], residual=x3))
        x = self.maxpool(x)
        x = F.relu(self.linear1(x))
        x = F.sigmoid(self.linear2(x))

        return x

class Discriminator(nn.Module):

    def __init__(self,
                 n_actions,
                 image_shape,
                 input_size,
                 output_size,
                 hidden_size=100):
        super().__init__()
        self.image_shape = image_shape

        self.conv1 = ConvBlock(3, 16, 8, padding=1)
        self.conv2 = ConvBlock(16, 32, 3, padding=1)
        self.conv3 = ConvBlock(32, 64, 3)
        self.conv4 = ConvBlock(64, 64, 3)
        self.conv5 = ConvBlock(64, 64, 3)
        self.maxpool = nn.MaxPool2d(2, 2, padding=1)

        self.linear1 = nn.Linear(16384, 100)  # a bit much ?
        self.linear2 = nn.Linear(100, n_actions)

        self.film_lstm = FiLM_LSTM(input_size, hidden_size, output_size)

    def forward(self, x, s):
        """
        x: input image;
        s: input sentence sequence (after coding, embedding/whatever)
        """
        gamma, beta = self.film_lstm(s)  # shape this

        x = F.relu(self.conv1(x, gamma[0], beta[0]))
        x = F.relu(self.conv2(x, gamma[1], beta[1]))
        x3 = F.relu(self.conv3(x, gamma[2], beta[2]))
        x = F.relu(self.conv4(x, gamma[3], beta[3]))
        x = F.relu(self.conv5(x, gamma[4], beta[4], residual=x3))
        x = self.maxpool(x)
        x = F.relu(self.linear1(x))
        x = F.softmax(self.linear2(x))

        return x