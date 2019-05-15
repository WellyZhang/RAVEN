import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from basic_model import BasicModel
from fc_tree_net import FCTreeNet

class conv_module(nn.Module):
    def __init__(self):
        super(conv_module, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        self.batch_norm4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(self.batch_norm1(x))
        x = self.conv2(x)
        x = self.relu2(self.batch_norm2(x))
        x = self.conv3(x)
        x = self.relu3(self.batch_norm3(x))
        x = self.conv4(x)
        x = self.relu4(self.batch_norm4(x))
        return x.view(-1, 16, 16*4*4)

class lstm_module(nn.Module):
    def __init__(self):
        super(lstm_module, self).__init__()
        self.lstm = nn.LSTM(input_size=16*4*4, hidden_size=96, num_layers=1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(96, 8)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        hidden, _ = self.lstm(x)
        score = self.fc(hidden[-1, :, :])
        return score

class CNN_LSTM(BasicModel):
    def __init__(self, args):
        super(CNN_LSTM, self).__init__(args)
        self.conv = conv_module()
        self.lstm = lstm_module()
        self.fc_tree_net = FCTreeNet(in_dim=300, img_dim=256)
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)

    def compute_loss(self, output, target, meta_target, meta_structure):
        pred = output[0]
        loss = F.cross_entropy(pred, target)
        return loss

    def forward(self, x, embedding, indicator):
        alpha = 1.0
        features = self.conv(x.view(-1, 1, 80, 80))
        features_tree = self.fc_tree_net(features, embedding, indicator)
        features_tree = features_tree.view(-1, 16, 256)
        final_features = features + alpha * features_tree
        score = self.lstm(final_features)
        return score, None

    