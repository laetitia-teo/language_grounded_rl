"""
This is the module for the reward and policy models.
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

    input_size, hidden_size are parameters for the LSTM layer.
    output_shapes describes the shapes of the output of the layer.
    it is a list of tuple of ints describing the different layer sizes
    """
    def __init__(self, 
                 input_size,
                 hidden_size,
                 output_shapes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)  # just one layer for now
        for i, output_shape in enumerate(output_shapes):
            n_flat = np.prod(output_shape[i])
            exp_g = 'self.gamma' + i + ' = nn.Linear(hidden_size, n_flat)'
            exp_b = 'self.beta' + i + ' = nn.Linear(hidden_size, n_flat)'
            eval(exp_g)
            eval(exp_b)

    def forward(self, x):
        """
        The output is a list of tuples containing gamma and beta for each
        convolutional layer.
        """
        # maybe provide initial hidden state and initial context
        output, (hn, cn) = self.lstm(x)
        hn = F.relu(hn)
        out = []
        for i, output_shape in enumerate(output_shapes):
            exp = 'out.append(self.gamma' + i + '(hn), self.beta' + i + '(hn))'
            eval(exp)
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

class Reward(nn.Module):

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

        output_shapes = [(33, 33, 16),
                         (31, 31, 32),
                         (31, 31, 64),
                         (31, 31, 64),
                         (31, 31, 64)]

        self.film_lstm = FiLM_LSTM(input_size, hidden_size, output_shapes)

    def forward(self, x, s):
        """
        x: input image;
        s: input sentence sequence (after coding, embedding/whatever)
        """
        out = self.film_lstm(s)  # shape this

        t0, t1, t2, t3, t4 = out
        for i in range(5):
            exp = 'gamma' + i + ', beta' + i + ' = t' + i
            eval(exp)

        x = F.relu(self.conv1(x, gamma0, beta0))
        x = F.relu(self.conv2(x, gamma1, beta1))
        x3 = F.relu(self.conv3(x, gamma2, beta2))
        x = F.relu(self.conv4(x, gamma3, beta3))
        x = F.relu(self.conv5(x, gamma4, beta4, residual=x3))
        x = self.maxpool(x)
        x = F.relu(self.linear1(x))
        x = F.sigmoid(self.linear2(x))

        return x

class Policy(nn.Module):

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

        output_shapes = [(33, 33, 16),
                         (31, 31, 32),
                         (31, 31, 64),
                         (31, 31, 64),
                         (31, 31, 64)]

        self.film_lstm = FiLM_LSTM(input_size, hidden_size, output_shapes)

    def forward(self, x, s):
        """
        x: input image;
        s: input sentence sequence (after coding, embedding/whatever)
        """
        out = self.film_lstm(s)  # shape this

        t0, t1, t2, t3, t4 = out
        for i in range(5):
            exp = 'gamma' + i + ', beta' + i + ' = t' + i
            eval(exp)

        x = F.relu(self.conv1(x, gamma[0], beta[0]))
        x = F.relu(self.conv2(x, gamma[1], beta[1]))
        x3 = F.relu(self.conv3(x, gamma[2], beta[2]))
        x = F.relu(self.conv4(x, gamma[3], beta[3]))
        x = F.relu(self.conv5(x, gamma[4], beta[4], residual=x3))
        x = self.maxpool(x)
        x = F.relu(self.linear1(x))
        x = F.softmax(self.linear2(x))

        return x