# -*- coding: utf-8 -*-
from dataLoaderSAE import GetLoaderSAE
import torch
import numpy as np
from torch.utils.data import DataLoader
# from model import MLP
from mymodel import MLP
from mymodel import AddGaussianNoise
import torch.optim as optim
import torch.nn as nn
import random

trainOnGpu = torch.cuda.is_available()
# trainOnGpu = False

if not trainOnGpu:
    print('CUDA is not available.')
else:

    print('CUDA is available!')

train_data_path = r'./mydata/trainData3_data.npy'
train_label_path = r'./mydata/trainData3_label.npy'

test_data_path = r'./mydata/testData3_data.npy'
test_label_path = r'./mydata/testData3_label.npy'

train_data = np.load(train_data_path, allow_pickle=True)
train_label = np.load(train_label_path, allow_pickle=True)
train_label = torch.tensor(train_label, dtype=torch.float32)

test_data = np.load(test_data_path, allow_pickle=True)
test_label = np.load(test_label_path, allow_pickle=True)
test_label = torch.tensor(test_label, dtype=torch.float32)

trainSAESpecs1 = train_data
trainLabels1 = train_label

testSAESpecs1 = test_data
testLabels1 = test_label
trainData1 = GetLoaderSAE(trainSAESpecs1, trainLabels1)
testData1 = GetLoaderSAE(testSAESpecs1, testLabels1)
print("trainData1:", trainData1)
print("testData1:", testData1)

batchSize = 8


trainLoader1 = torch.utils.data.DataLoader(trainData1, batch_size=batchSize, shuffle=True, drop_last=False, num_workers=0)
testLoader1 = torch.utils.data.DataLoader(testData1, batch_size=batchSize, shuffle=True, drop_last=False, num_workers=0)


model = MLP()
print(model)


Epochs = 100
for Epoch in range(1, Epochs + 1):

    if trainOnGpu:
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    epochs = 70
    testLossMin = np.Inf

    for epoch in range(1, epochs + 1):

        trainLoss = 0.0

        model.train()
        # with torch.no_grad():
        for data, target in trainLoader1:

            if trainOnGpu:
                data, target = data.cuda().type(torch.cuda.FloatTensor), target.cuda().type(torch.cuda.FloatTensor)

            data = data.squeeze(dim=1)
            data = torch.tensor(data, dtype=torch.float32)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.long()).mean()
            loss.backward()
            optimizer.step()

            trainLoss += loss.item() * data.size(0)

        trainLoss = trainLoss / len(trainLoader1.sampler)

        print('epoch: {}'.format(epoch), 'trainLoss: {}'.format(trainLoss))

    model.eval()
    testLoss = 0.0
    total_correct = 0
    total_itm = 0
    classCorrect = list(0. for i in range(6))
    classTotal = list(0. for i in range(6))

    for data, target in testLoader1:
        with torch.no_grad():
            if trainOnGpu:
                data, target = data.cuda().type(torch.cuda.FloatTensor), target.cuda().type(torch.cuda.FloatTensor)

            data = data.squeeze(dim=1)
            data = data.cpu().numpy()
            data = AddGaussianNoise(mean=random.uniform(0.5, 1.5), variance=0.5, amplitude=random.uniform(0, 45))(data)
            data = torch.tensor(data, dtype=torch.float32)
            data = data.cuda().type(torch.cuda.FloatTensor)

            output = model(data)

            loss = criterion(output, target.long())

            testLoss += loss.item() * data.size(0)

            zero = torch.zeros_like(output)
            one = torch.ones_like(output)
            pred1 = torch.where(output > 0.167, one, output)
            pred1 = torch.where(output < 0.167, zero, pred1)

            dim0, dim1 = pred1.shape
            pred = []
            for i in range(dim0):
                maxIdxes, = torch.where(pred1[i] == pred1[i].max())
                pred.append(np.random.choice(maxIdxes.cpu().numpy()))

            pred = torch.tensor(pred, dtype=torch.float32)
            pred = pred.cuda().type(torch.cuda.FloatTensor)

            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not trainOnGpu else np.squeeze(correct_tensor.cpu().numpy())

            total_correct += correct.sum()
            total_itm += correct.shape[0]

    testLoss = testLoss / len(trainLoader1.dataset)

    print('Epoch: {}'.format(Epoch), 'Test Accuracy (Overall): %.2f ' % (
        total_correct / total_itm))

