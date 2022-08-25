import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init
from torch.nn.functional import elu

import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import math
from braindecode.models.modules import Expression
from braindecode.util import np_to_var
from braindecode.models.functions import safe_log, square

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


class CNN2D(torch.nn.Module):
    def __init__(self, input_size, kernel_size, conv_channels,
                 dense_size, nclass, dropout):
        super(CNN2D, self).__init__()
        self.cconv = []
        self.MaxPool = nn.MaxPool2d((2, 1), (2, 1))
        self.ReLU = nn.ReLU()
        self.Dropout = nn.Dropout(dropout)
        self.batchnorm = []
        # ############ batchnorm ###########
        # for jj in conv_channels:
        #     self.batchnorm.append(nn.BatchNorm2d(jj, eps=0.001, momentum=0.01,
        #                                          affine=True, track_running_stats=True).cuda())
        ii = 0  # define CONV layer architecture: #####
        for in_channels, out_channels in zip(conv_channels, conv_channels[1:]):
            conv_i = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size[ii],  # stride = (1, 2),
                                     padding=(kernel_size[ii][0] // 2, kernel_size[ii][1] // 2))
            self.cconv.append(conv_i)
            self.add_module('CNN_K{}_O{}'.format(kernel_size[ii], out_channels), conv_i)
            ii += 1

        #self.fc_conv = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 4), padding=(1, 2))
        #self.fc_pool = nn.MaxPool2d((3, 4))
        #self.fc_drop = nn.Dropout(0.5)
        #self.fc_conv1 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 4), padding=(1, 2))
        #self.fc_pool1 = nn.MaxPool2d((3, 4))
        #self.fc_drop1 = nn.Dropout(0.5)
        self.flat_dim = self.get_output_dim(input_size, self.cconv)
        #self.flat_dim = 3040
        self.fc1 = torch.nn.Linear(self.flat_dim, dense_size)
        self.fc2 = torch.nn.Linear(dense_size, nclass)

    def get_output_dim(self, input_size, cconv):
        with torch.no_grad():
            input = torch.ones(1, *input_size)
            for conv_i in cconv:
                input = self.MaxPool(conv_i(input))
                flatout = int(np.prod(input.size()[1:]))
                print("Input shape : {} and flattened : {}".format(input.shape, flatout))
        return flatout

    def forward(self, x):
        for jj, conv_i in enumerate(self.cconv):
            x = conv_i(x)
            # x = self.batchnorm[jj + 1](x)
            x = self.ReLU(x)
            x = self.MaxPool(x)

        # flatten the CNN output
        #x = F.relu(self.fc_conv(x))
        #x = self.fc_pool(self.fc_drop(x))
        # print(x.shape)
        out = x.view(-1, self.flat_dim)
        out = F.relu(self.fc1(out))
        out = self.Dropout(out)
        out = self.fc2(out)
        return out


class CNN_LSTM(torch.nn.Module):
    def __init__(self, input_size, kernel_size, conv_channels,
                 dense_size, dropout, nclass=3):
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
        ii = 0  # define CONV layer architecture: #####
        for in_channels, out_channels in zip(conv_channels, conv_channels[1:]):
            conv_i = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size[ii], stride=(2, 1),
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
        output = input.view(len(input), shape[1] * shape[2], -1)
        return output.shape

    def get_output_dim(self, input_size, cconv):
        with torch.no_grad():
            input = torch.ones(1, *input_size)
            for conv_i in cconv:
                input = self.MaxPool(conv_i(input))
                flatout = int(np.prod(input.size()[1:]))
                print("Input shape : {} and flattened : {}".format(input.shape, flatout))
        shape = input.shape
        output = input.view(len(input), shape[1] * shape[2], -1)
        print("LSTM input: ", output.shape)
        h_lstm, _ = self.lstm(output)
        print("output LSTM: ", output.shape)
        output = torch.flatten(h_lstm, 1)
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
        output = x.view(len(x), shape[1] * shape[2], -1)
        h_lstm, _ = self.lstm(output)
        output = torch.flatten(h_lstm, 1)
        output = F.relu(self.fc1(output))
        output = self.Dropout(output)
        output = self.fc2(output)
        return output

class GraphConvolution(nn.Module):
    """
    Graph Convolution layer
    """

    def __init__(self, in_dim: int, out_dim: int, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.Tensor(16, in_dim, out_dim))

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
                 dense_size, dropout, nclass=3):
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
        output = input.view(len(input), shape[1] * shape[2], -1)
        print("GCN output: ", output.shape)
        # h_lstm, _ = self.lstm(output)
        # print("output LSTM: ", output.shape)
        output = torch.flatten(output, 1)
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
        output = x.view(len(x), -1)
        # h_lstm, _ = self.lstm(output)
        # output = torch.flatten(h_lstm,1)
        output = F.relu(self.fc1(output))
        output = self.Dropout(output)
        output = self.fc2(output)
        return output


def base_model(in_chans):
    fixmodel = CreateBase(in_chans)
    return fixmodel


def _transpose_time_to_spat(x):
    return x.permute(0, 3, 2, 1)


def _transpose_spat_to_time(x):
    x = x.permute(0, 3, 2, 1)
    return x.contiguous()


def _squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


class CreateBase(nn.Module):
    def __init__(self, in_chans,
                 n_filters_time=10,
                 n_filters_spat=40,
                 filter_time_length=25,
                 pool_time_length=75,
                 pool_time_stride=15,
                 conv_nonlin=square,
                 pool_nonlin=safe_log,
                 drop_prob=0.75,
                 batch_norm=True,
                 batch_norm_alpha=0.1):
        super(CreateBase, self).__init__()

        #       block1

        self.conv_time = nn.Conv2d(1, n_filters_time,
                                   (filter_time_length, 1),
                                   stride=1, )
        init.xavier_uniform_(self.conv_time.weight, gain=1)
        init.constant_(self.conv_time.bias, 0)
        self.conv_spat = nn.Conv2d(n_filters_time, n_filters_spat,
                                   (1, in_chans),
                                   stride=(1, 1),
                                   bias=not batch_norm)
        init.xavier_uniform_(self.conv_spat.weight, gain=1)
        self.bnorm = nn.BatchNorm2d(n_filters_spat,
                                    momentum=batch_norm_alpha,
                                    affine=True)
        init.constant_(self.bnorm.weight, 1)
        init.constant_(self.bnorm.bias, 0)
        self.conv_nonlin = Expression(conv_nonlin)
        self.pool = nn.AvgPool2d(
            kernel_size=(pool_time_length, 1),
            stride=(pool_time_stride, 1))
        self.pool_nonlin = Expression(pool_nonlin)
        self.drop = nn.Dropout(p=drop_prob)

    def forward(self, data):
        data = self.conv_spat(self.conv_time(_transpose_time_to_spat(data)))
        data = self.pool(self.conv_nonlin(self.bnorm(data)))
        data = self.drop(self.pool_nonlin(data))
        return data


class EEGClassifier(nn.Module):

    def __init__(self, n_out_time, filtersize=50, n_classes=4):
        super(EEGClassifier, self).__init__()

        self.conv_class_s = nn.Conv2d(filtersize, n_classes,
                                      (n_out_time, 1), bias=True)
        init.xavier_uniform_(self.conv_class_s.weight, gain=1)
        init.constant_(self.conv_class_s.bias, 0)
        self.softmax_s = nn.LogSoftmax(dim=1)
        self.squeeze_s = Expression(_squeeze_final_output)

    def forward(self, data):
        return self.squeeze_s(self.softmax_s(self.conv_class_s(data)))


class EEGShallowClassifier(nn.Module):
    def __init__(self, in_chans, n_classes, input_time_length, return_feature=False, reductionsize=10, cat_features=0, if_reduction=False, if_deep=False):
        super(EEGShallowClassifier, self).__init__()
        self.basenet = base_model(in_chans)
        x0 = np_to_var(np.ones(
            (1, in_chans, input_time_length, 1),
            dtype=np.float32))
        # x0 = self.basenet(x0)
        # n_out = x0.cpu().data.numpy().shape
        # filtersize = n_out[1]
        # n_out_time = n_out[2]
        filtersize, n_out_time = 40, 2
#         print('feature shape is: ', n_out)

        if if_reduction:
            self.feature_reduction = nn.Conv2d(filtersize, reductionsize,
                                               (n_out_time, 1), bias=True)
            init.xavier_uniform_(self.feature_reduction.weight, gain=1)
            init.constant_(self.feature_reduction.bias, 0)
            if if_deep:
                self.deep1 = nn.Conv2d(reductionsize, reductionsize,
                                       (1, 1), bias=True)
                init.xavier_uniform_(self.deep1.weight, gain=1)
                init.constant_(self.deep1.bias, 0)

                self.deep2 = nn.Conv2d(reductionsize, reductionsize,
                                       (1, 1), bias=True)
                init.xavier_uniform_(self.deep2.weight, gain=1)
                init.constant_(self.deep2.bias, 0)

                self.deep3 = nn.Conv2d(reductionsize, reductionsize,
                                       (1, 1), bias=True)
                init.xavier_uniform_(self.deep3.weight, gain=1)
                init.constant_(self.deep3.bias, 0)
                x0 = self.deep3(self.deep2(self.deep1(self.feature_reduction(x0))))
            else:
                x0 = self.feature_reduction(x0)
            n_out = x0.cpu().data.numpy().shape
#             print('feature reduction shape is: ', n_out)
            self.classifier = EEGClassifier(n_out_time=1,
                                            n_classes=n_classes, filtersize=reductionsize + cat_features)
        else:
            self.classifier = EEGClassifier(n_out_time=n_out_time,
                                            n_classes=n_classes, filtersize=filtersize + cat_features)

        self.return_feature = return_feature
        self.if_reduction = if_reduction
        self.if_deep = if_deep

    def forward(self, data, cat_feature=None):
        feature = self.basenet(data)
        if self.return_feature:
            return feature
        if self.if_reduction:
            if self.if_deep:
                feature = self.deep3(self.deep2(self.deep1(self.feature_reduction(feature))))
            else:
                feature = self.feature_reduction(feature)
            if cat_feature is not None:
                feature = torch.cat((feature, cat_feature), 1)
            y = self.classifier(feature)
            return y
        else:
            if cat_feature is not None:
                feature = torch.cat((feature, cat_feature), 1)
            y = self.classifier(feature)
            return y
