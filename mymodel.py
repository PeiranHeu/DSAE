# -*- coding: utf-8 -*-

import torch
import swin_transformer as st
import torch.nn as nn
from decoderBlock import DecoderBlock
import random
import numpy as np


class SAE(nn.Module):
    def __init__(self, BN_enable=True, resNet_pretrain=False):
        super(SAE, self).__init__()
        self.BN_enable = BN_enable
        filters = [64, 256, 512, 1024, 2048]

        self.firstConv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1)
        self.Conv1 = nn.Conv1d(in_channels=16, out_channels=64, kernel_size=1, stride=1)
        self.Conv2 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=1, stride=1)
        self.Conv3 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=1, stride=1)
        self.Conv4 = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=1, stride=1)
        self.lastConv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

        self.model = st.swin_base_patch4_window7_224()


        self.center = DecoderBlock(in_channels=filters[4], mid_channels=filters[4] * 4, out_channels=filters[4],
                                   BN_enable=self.BN_enable)
        self.decoder1 = DecoderBlock(in_channels=filters[4] + filters[3], mid_channels=filters[3] * 4,
                                     out_channels=filters[3], BN_enable=self.BN_enable)
        self.decoder2 = DecoderBlock(in_channels=filters[3] + filters[2], mid_channels=filters[2] * 4,
                                     out_channels=filters[2], BN_enable=self.BN_enable)
        self.decoder3 = DecoderBlock(in_channels=filters[2] + filters[1], mid_channels=filters[1] * 4,
                                     out_channels=filters[1], BN_enable=self.BN_enable)
        self.decoder4 = DecoderBlock(in_channels=filters[1] + filters[0], mid_channels=filters[0] * 4,
                                     out_channels=filters[0], BN_enable=self.BN_enable)

        if self.BN_enable:
            self.final = nn.Sequential(
                nn.Conv2d(in_channels=filters[0], out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.final = nn.Sequential(
                nn.Conv2d(in_channels=filters[0], out_channels=32, kernel_size=3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
                nn.Sigmoid()
            )

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.firstConv(x)
        y = self.lastConv(x)
        outputs = self.model(x)

        e1 = outputs[0]
        e2 = outputs[1]
        e3 = outputs[2]
        e4 = outputs[3]

        return e4


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(1024, 512)
        nn.init.kaiming_uniform_(self.hidden1.weight, nonlinearity="relu")
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(512, 256)
        nn.init.kaiming_uniform_(self.hidden2.weight, nonlinearity="relu")
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(256, 128)
        nn.init.kaiming_uniform_(self.hidden3.weight, nonlinearity="relu")
        self.act3 = nn.ReLU()
        self.hidden4 = nn.Linear(128, 64)
        nn.init.kaiming_uniform_(self.hidden4.weight, nonlinearity="relu")
        self.act4 = nn.ReLU()
        self.hidden5 = nn.Linear(64, 32)
        nn.init.kaiming_uniform_(self.hidden5.weight, nonlinearity="relu")
        self.act5 = nn.ReLU()
        self.hidden6 = nn.Linear(32, 6)
        nn.init.xavier_uniform_(self.hidden6.weight)
        self.act6 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.act1(x)
        x = self.hidden2(x)
        x = self.act2(x)
        x = self.hidden3(x)
        x = self.act3(x)
        x = self.hidden4(x)
        x = self.act4(x)
        x = self.hidden5(x)
        x = self.act5(x)
        x = self.hidden6(x)
        y = self.act6(x)
        return y


class MLR(nn.Module):
    def __init__(self):
        super(MLR, self).__init__()
        self.linear1 = nn.Linear(2048, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 32)
        self.linear4 = nn.Linear(32, 8)
        self.linear5 = nn.Linear(8, 2)
        self.linear6 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        x = self.sigmoid(self.linear5(x))
        y = self.sigmoid(self.linear6(x))
        return y


class MyMLP(nn.Module):
    def __init__(self, model1=SAE(), model2=MLP()):
        super(MyMLP, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        x = self.model1(x)
        x = x.squeeze(dim=1)
        x = self.model2(x)

        return x


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(1024, 512)
        nn.init.kaiming_uniform_(self.hidden1.weight, nonlinearity="relu")
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(512, 256)
        nn.init.kaiming_uniform_(self.hidden2.weight, nonlinearity="relu")
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(256, 128)
        nn.init.kaiming_uniform_(self.hidden3.weight, nonlinearity="relu")
        self.act3 = nn.ReLU()
        self.hidden4 = nn.Linear(128, 64)
        nn.init.kaiming_uniform_(self.hidden4.weight, nonlinearity="relu")
        self.act4 = nn.ReLU()
        self.hidden5 = nn.Linear(64, 32)
        nn.init.kaiming_uniform_(self.hidden5.weight, nonlinearity="relu")
        self.act5 = nn.ReLU()
        self.hidden6 = nn.Linear(32, 6)
        nn.init.xavier_uniform_(self.hidden6.weight)
        self.act6 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.act1(x)
        x = self.hidden2(x)
        x = self.act2(x)
        x = self.hidden3(x)
        x = self.act3(x)
        x = self.hidden4(x)
        x = self.act4(x)
        x = self.hidden5(x)
        x = self.act5(x)
        x = self.hidden6(x)
        y = self.act6(x)
        return y


class MLR(nn.Module):
    def __init__(self):
        super(MLR, self).__init__()
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 6)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.relu(self.linear2(x))
        return x


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0, p=1):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p = p

    def __call__(self, data):
        if random.uniform(0, 1) < self.p:
            data = np.array(data)
            h, w = data.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w))
            # N = np.repeat(N, axis=2)
            N = (N - np.min(N)) / (np.max(N) - np.min(N))
            data = N + data
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
            return data
        else:
            return data


if __name__ == "__main__":
    model = SAE()

    x = torch.rand(8, 1, 32, 32)

    y1 = model(x)
    print(y1.shape)
