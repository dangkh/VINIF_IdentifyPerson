import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler

device = 'cuda' if torch.cuda.is_available() else 'cpu'

h1, w1 = 3, 1
h2, w2 = 3, 3
h3, w3 = 3, 5
width = 10
height = 10
paramsCNN2D = {'conv_channels': [
            [1, 16, 8],
            [1, 16, 32],
            [1, 64, 32, 16, 8],
            [1, 128, 64, 32, 16, 8],
            [1, 256, 128, 64, 32, 16, 8]
        ],
            'kernel_size': [[(h1, w1 * width), (h1, w1 * width), (h1, w1 * width),
                             (h1, w1 * width), (h1, w1 * width), (h1, w1 * width)],

                            [(h2 * height, w2), (h2 * height, w2), (h2 * height, w2),
                             (h2 * height, w2), (h2 * height, w2), (h2 * height, w2)],

                            [(h3, w3 * width), (h3, w3 * width), (h3, w3 * width),
                             (h3, w3 * width), (h3, w3 * width), (h3, w3 * width)]]
        }

class CNN_LSTM(torch.nn.Module):
    def __init__(self, input_size, kernel_size, conv_channels,
                 dense_size, dropout, nclass = 3):
        super(CNN_LSTM, self).__init__()
        self.cconv = []
        self.MaxPool = nn.MaxPool2d((2, 1), (2, 1))
        self.ReLU = nn.ReLU()
        self.Dropout = nn.Dropout(dropout)
        self.batchnorm = []
        self.bn = nn.BatchNorm2d(32, momentum=0.1, affine=True)
        # ############ batchnorm ###########
        for jj in conv_channels:
            self.batchnorm.append(nn.BatchNorm2d(jj, eps=0.001, momentum=0.01,
                                                 affine=True))
        ii = 0  ##### define CONV layer architecture: #####
        for in_channels, out_channels in zip(conv_channels, conv_channels[1:]):
            conv_i = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size[ii],  stride = (2, 1),
                                     padding=(kernel_size[ii][0] // 2, kernel_size[ii][1] // 2))
            self.cconv.append(conv_i)
            self.add_module('CNN_K{}_O{}'.format(kernel_size[ii], out_channels), conv_i)
            ii += 1

        output_LSTM = self.get_output_LSTMdim(input_size, self.cconv)
        numFeatures = output_LSTM[-1]
        self.lstm = nn.LSTM(numFeatures, 8, bidirectional=True, batch_first=True)
        self.flat_dim = self.get_output_dim(input_size, self.cconv)
        self.fc1 = torch.nn.Linear(self.flat_dim, dense_size)
        self.fc2 = torch.nn.Linear(dense_size, nclass)

    def get_output_LSTMdim(self, input_size, cconv):
        with torch.no_grad():
            input = torch.ones(1, *input_size)
            for conv_i in cconv:
                input = self.MaxPool(conv_i(input))
                flatout = int(np.prod(input.size()[1:]))
        shape = input.shape
        output = input.view(len(input), shape[1] * shape[2] ,-1)
        return output.shape

    def get_output_dim(self, input_size, cconv):
        with torch.no_grad():
            input = torch.ones(1, *input_size)
            for conv_i in cconv:
                input = self.MaxPool(conv_i(input))
                flatout = int(np.prod(input.size()[1:]))
                print("Input shape : {} and flattened : {}".format(input.shape, flatout))
        shape = input.shape
        output = input.view(len(input), shape[1] * shape[2] ,-1)
        print("LSTM input: ", output.shape)
        h_lstm, _ = self.lstm(output)
        print("output LSTM: ", output.shape)
        output = torch.flatten(h_lstm,1)
        return output.shape[-1]

    def forward(self, x):
        for jj, conv_i in enumerate(self.cconv):
            # print(jj)
            x = conv_i(x)
            if jj == len(self.cconv) - 1:
                x = self.bn(x)
            x = self.ReLU(x)
            x = self.MaxPool(x)
        shape = x.shape
        output = x.view(len(x), shape[1] * shape[2] ,-1)
        h_lstm, _ = self.lstm(output)
        output = torch.flatten(h_lstm,1)
        output = F.relu(self.fc1(output))
        output = self.Dropout(output)
        output = self.fc2(output)
        return output
