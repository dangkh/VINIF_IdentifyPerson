import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler


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


class LSTMNet(nn.Module):
    def __init__(self, n_classes):
        super(LSTMNet, self).__init__()
        self.drop1 = nn.Dropout(0.75)
        self.drop2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(60000, 2)
        self.n_classes = n_classes
        self.fc2 = nn.Linear(2, n_classes)
        self.conv = nn.Conv2d(1, 32, [3, 10])
        self.lstm = nn.LSTM(28, 15, bidirectional=True, batch_first=True)

    def forward(self, x):
        output = self.conv(x)
        output =  F.relu(output)
        shape = output.shape
        output = output.view(len(output), shape[1] * shape[2] ,-1)
        h_lstm, _ = self.lstm(output)
        output = torch.flatten(h_lstm,1)
        output = self.drop1(output)
        output = self.fc1(output)
        output = self.fc2(output)
        scores = F.log_softmax(output, dim=-1)
        return scores

    def get_embedding(self, x):
        output = self.conv(x)
        output =  F.relu(output)
        shape = output.shape
        output = output.view(len(output), shape[1] * shape[2] ,-1)
        h_lstm, _ = self.lstm(output)
        output = torch.flatten(h_lstm,1)
        output = self.drop1(output)
        output = self.fc1(output)
        return output


class CNN_LSTM(torch.nn.Module):
    def __init__(self, input_size, kernel_size, conv_channels,
                 dense_size, dropout, nclass = 3):
        super(CNN_LSTM, self).__init__()
        self.cconv = []
        self.MaxPool = nn.MaxPool2d((1, 2), (1, 2))
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

class TripletNet(nn.Module):
    def __init__(self, baseNet):
        super().__init__()
        self.embedding_net = baseNet

    def forward(self, x1):
        output1 = self.embedding_net(x1)
        return output1

    def get_embedding(self, x):
        return self.embedding_net(x)

# Compute the distance matrix
# https://omoindrot.github.io/triplet-loss
# numpy + torch version
def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = torch.matmul(embeddings, embeddings.T)

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = torch.unsqueeze(square_norm, 0) - 2.0 * dot_product + torch.unsqueeze(square_norm, 1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = torch.maximum(distances, torch.tensor(0.0))

    if not squared:
        # # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # # we need to add a small epsilon where distances == 0.0
        # mask = tf.to_float(tf.equal(distances, 0.0))
        # distances = distances + mask * 1e-16

        distances = torch.sqrt(distances)

        # # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        # distances = distances * (1.0 - mask)

    return distances

def _get_triplet_mask(labels):
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    nn = len(labels)
    mask = torch.ones([nn, nn, nn])
    for a in range(nn):
        for p in range(nn):
            for n in range(nn):
                if labels[a] != labels[p] or labels[n] == labels[a] or a == p:
                    mask[a, p, n] = 0
    return mask

# Compute the distance matrix
# https://omoindrot.github.io/triplet-loss
# numpy + torch version

def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)
    anchor_positive_dist = torch.unsqueeze(pairwise_dist, 2)
    anchor_negative_dist = torch.unsqueeze(pairwise_dist, 1)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = torch.as_tensor(_get_triplet_mask(labels)).requires_grad_(False).to(device)
    triplet_loss = torch.mul(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = torch.maximum(triplet_loss, torch.tensor(0.0))
    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = torch.gt(triplet_loss, 1e-16)
    num_positive_triplets = torch.sum(valid_triplets)
    num_valid_triplets = torch.sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)
    return triplet_loss, fraction_positive_triplets