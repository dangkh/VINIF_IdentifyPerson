import torch.nn as nn
import torch.nn.functional as F
    
class LSTMNet(nn.Module):
    def __init__(self, n_classes):
        super(LSTMNet, self).__init__()
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(60000, 2)
        self.n_classes = n_classes
        self.fc2 = nn.Linear(2, n_classes)
        self.conv = nn.Conv2d(1, 16, [3, 10])
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