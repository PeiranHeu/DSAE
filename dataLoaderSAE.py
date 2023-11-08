# -*- coding: utf-8 -*-
import torch

class GetLoaderSAE(torch.utils.data.Dataset):
    def __init__(self, data, target):
        self.data = data
        self.label = target

    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    def __len__(self):
        return len(self.data)