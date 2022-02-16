import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import math

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

# class GCN_Cuong(nn.Module):
#     """
#     Simple Graph Convolutional Networks for Node Embedding with 2 Graph Convolution layers
#     """

#     def __init__(self, in_dim, hidden_dim, out_dim, dropout, num_GC_layers=2):
#         """
#         @param in_dim: dimension of input features
#         @param hidden_dim: dimension of hidden layer
#         @param out_dim: dimension of output
#         @param num_GC_layers: number of Graph Convolution layers with value: 1 or 2, default 2
#         @param dropout: dropout layer
#         """
#         super(GCN, self).__init__()
#         self.in_dim = in_dim
#         self.hidden_dim = hidden_dim
#         self.out_dim = out_dim
#         self.num_GC_layers = num_GC_layers
#         self.dropout = dropout
#         if self.num_GC_layers == 1:
#             self.gc1 = GraphConvolution(in_dim, out_dim)
#         if self.num_GC_layers == 2:
#             self.gc1 = GraphConvolution(in_dim, hidden_dim)
#             self.gc2 = GraphConvolution(hidden_dim, out_dim)
#         self.fc = nn.Linear(out_dim * 2, out_dim)

#     def forward(self, x, adj, nodes_u, nodes_v):
#         """
#         @param x: node features
#         @param adj: adjacency matrix
#         @return: node embedding
#         """
#         h = F.relu(self.gc1(x, adj))
#         h = F.dropout(h, self.dropout, training=self.training)
#         if self.num_GC_layers == 2:
#             h = F.relu(self.gc2(h, adj))
#             h = F.dropout(h, self.dropout, training=self.training)
#         h_uv = torch.cat((h[nodes_u], h[nodes_v]), 1)
#         scores = self.fc(h_uv)
#         return scores

class GraphConvolution(nn.Module):
    """
    Graph Convolution layer
    """

    def __init__(self, in_dim: int, out_dim: int, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.Tensor(8, in_dim, out_dim))
        

        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        s1, s2, s3, s4 = input.shape
        newInp = input.reshape(s1, s2, s4, s3)
        # print("debug shape")
        # print(adj.shape)
        # print(newInp.shape)
        # print(self.weight.shape)
        support = torch.matmul(newInp, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            output = output + self.bias
        s1, s2, s3, s4 = output.shape
        output = output.reshape(s1, s2, s4, s3)
        # print(output.shape)
        # print('end')

        return output
        # return input

    def __repr__(self):
        return '{}(in_dim={}, out_dim={})'.format(self.__class__.__name__, self.in_dim, self.out_dim)


class GCN(torch.nn.Module):
    def __init__(self, input_size, kernel_size, conv_channels,
                 dense_size, dropout, nclass = 3):
        super(GCN, self).__init__()
        
        self.gcn1 = GraphConvolution(128, 64)
        self.gcn2 = GraphConvolution(64, 16)
        # output_LSTM = self.get_output_LSTMdim(input_size, self.gcn1)
        # numFeatures = 1
        # # numFeatures = output_LSTM[-1]
        # self.lstm = nn.LSTM(numFeatures, 8, bidirectional=True, batch_first=True)
        self.flat_dim = self.get_output_dim(input_size)
        # self.flat_dim = 128*32
        self.Dropout = nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(self.flat_dim, dense_size)
        self.fc2 = torch.nn.Linear(dense_size, nclass)


    def get_output_dim(self, input_size):
        with torch.no_grad():
            input = torch.ones(1, *input_size)
        tmpAdj = torch.Tensor(32, 32)
        print(input.shape)
        input = self.gcn1(input, tmpAdj)
        print(input.shape)
        input = self.gcn2(input, tmpAdj)
        flatout = int(np.prod(input.size()[1:]))
        print("Input shape : {} and flattened : {}".format(input.shape, flatout))
        shape = input.shape
        output = input.view(len(input), shape[1] * shape[2] ,-1)
        print("GCN output: ", output.shape)
        # h_lstm, _ = self.lstm(output)
        # print("output LSTM: ", output.shape)
        output = torch.flatten(output,1)
        return output.shape[-1]

    # def get_output_LSTMdim(self, input_size, gcn):
    #     with torch.no_grad():
    #         input = torch.ones(1, *input_size)
    #     tmpAdj = torch.Tensor(32, 32)
    #     input = self.gcn1(input, tmpAdj)
    #     print(input.shape)
    #     flatout = int(np.prod(input.size()[1:]))
    #     print("Input shape : {} and flattened : {}".format(input.shape, flatout))
    #     shape = input.shape
    #     output = input.view(len(input), shape[1] * shape[2] ,-1)
    #     print("LSTM input: ", output.shape)
    #     h_lstm, _ = self.lstm(output)
    #     print("output LSTM: ", output.shape)
    #     output = torch.flatten(h_lstm,1)
    #     print(output.shape)
    #     stop
    #     return output.shape[-1]

    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj))
        x = self.Dropout(x)
        x = self.gcn2(x, adj)
        shape = x.shape
        output = x.view(len(x),-1)
        # h_lstm, _ = self.lstm(output)
        # output = torch.flatten(h_lstm,1)
        output = F.relu(self.fc1(output))
        output = self.Dropout(output)
        output = self.fc2(output)
        return output
