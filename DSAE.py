# -*- coding: utf-8 -*-
from dataLoaderSAE import GetLoaderSAE
from pickle import *
import torch
import numpy as np
from torch.utils.data import DataLoader
from mymodel import SAE
import torch.optim as optim
import torch.nn as nn
import memory
import torch.nn.functional as F
import tqdm


trainOnGpu = torch.cuda.is_available()

if not trainOnGpu:
    print('CUDA is not available.')
else:
    print('CUDA is available!')


trainF1 = open("./data/trainData1", "rb")
trainF2 = open("./data/trainData2", "rb")
testF1 = open("./data/testData1", "rb")
testF2 = open("./data/testData2", "rb")
trainDt1 = load(trainF1, encoding="latin1")
trainDt2 = load(trainF2, encoding="latin1")
testDt1 = load(testF1, encoding="latin1")
testDt2 = load(testF2, encoding="latin1")

trainFBanks1 = trainDt1["FBanks"].reshape(-1, 1, 32, 32)
trainGBanks1 = np.array(trainDt1["GBanks"])
trainLabels1 = np.array(trainDt1["labels"])

trainFBanks2 = trainDt2["FBanks"].reshape(-1, 1, 32, 32)
trainGBanks2 = np.array(trainDt2["GBanks"])
trainLabels2 = np.array(trainDt2["labels"])

testFBanks1 = testDt1["FBanks"].reshape(-1, 1, 32, 32)
testGBanks1 = np.array(testDt1["GBanks"])
testLabels1 = np.array(testDt1["labels"])

testFBanks2 = testDt2["FBanks"].reshape(-1, 1, 32, 32)
testGBanks2 = np.array(testDt2["GBanks"])
testLabels2 = np.array(testDt2["labels"])

trainData1 = GetLoaderSAE(trainFBanks1, trainGBanks1)
trainData2 = GetLoaderSAE(trainFBanks2, trainGBanks2)

trainData3 = GetLoaderSAE(trainFBanks1, trainLabels1)
trainData4 = GetLoaderSAE(trainFBanks2, trainLabels2)

testData1 = GetLoaderSAE(testFBanks1, testGBanks1)
testData2 = GetLoaderSAE(testFBanks2, testGBanks2)

testData3 = GetLoaderSAE(testFBanks1, testLabels1)
testData4 = GetLoaderSAE(testFBanks2, testLabels2)


batchSize = 1


trainLoader1 = torch.utils.data.DataLoader(trainData1, batch_size=batchSize, shuffle=True, drop_last=False,
                                           num_workers=0)
trainLoader2 = torch.utils.data.DataLoader(trainData2, batch_size=batchSize, shuffle=True, drop_last=False,
                                           num_workers=0)
trainLoader3 = torch.utils.data.DataLoader(trainData3, batch_size=batchSize, shuffle=True, drop_last=False,
                                           num_workers=0)
trainLoader4 = torch.utils.data.DataLoader(trainData4, batch_size=batchSize, shuffle=True, drop_last=False,
                                           num_workers=0)
testLoader1 = torch.utils.data.DataLoader(testData1, batch_size=batchSize, shuffle=True, drop_last=False, num_workers=0)
testLoader2 = torch.utils.data.DataLoader(testData2, batch_size=batchSize, shuffle=True, drop_last=False, num_workers=0)
testLoader3 = torch.utils.data.DataLoader(testData3, batch_size=batchSize, shuffle=True, drop_last=False, num_workers=0)
testLoader4 = torch.utils.data.DataLoader(testData4, batch_size=batchSize, shuffle=True, drop_last=False, num_workers=0)


model = SAE()

print(model)

if trainOnGpu:
    model.cuda()


criterion = nn.CosineEmbeddingLoss(margin=0.2)
dumLabel = torch.ones(batchSize)
wrongLabel = torch.zeros(batchSize)
posLabel = torch.ones(batchSize * 8)
if trainOnGpu:
    wrongLabel = wrongLabel.cuda()
    posLabel = posLabel.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
punish_a = 0.1

testLossMin = np.Inf

neg_memory_size = 512
neg_memory = memory.Memory()
pos_memory_size = 200
pos_memory_length = 8
pos_memory = memory.PosMemory()


for epoch in range(1, epochs + 1):


    trainLoss = 0.0
    testLoss = 0.0

    model.train()

    count = 0
    tqdm_trainLoader1 = tqdm.tqdm(trainLoader1)
    for data, target in tqdm_trainLoader1:

        if trainOnGpu:
            data, target, dumLabel = data.cuda().type(torch.cuda.FloatTensor), target.cuda().type(
                torch.cuda.FloatTensor), dumLabel.cuda().type(torch.cuda.FloatTensor)

        optimizer.zero_grad()
        target = target.unsqueeze(dim=1)
        output1 = model(data).reshape(batchSize, -1)
        output2 = model(target).reshape(batchSize, -1)
        if count < pos_memory_size:
            loss3 = pos_memory.in_memory(output2, count, batchSize)

        neg_memory.in_memory(output1)
        neg_feature = torch.cat(neg_memory.memory, dim=0).reshape(neg_memory_size, -1)

        loss1 = criterion(output1, output2, dumLabel)
        loss2 = F.normalize(output1, p=2, dim=1) @ F.normalize(output2, p=2, dim=1).t()

        for i in range(batchSize):
            loss2[i][i] = 0

        loss = loss1 + (loss2.mean() + loss3) * punish_a
        loss.backward(retain_graph=True)
        optimizer.step()
        trainLoss += loss.item() * data.size(0)
        count += batchSize


    model.eval()
    tqdm_testLoader1 = tqdm.tqdm(testLoader1)
    for data, target in tqdm_testLoader1:

        if trainOnGpu:
            data, target, dumLabel = data.cuda().type(torch.cuda.FloatTensor), target.cuda().type(
                torch.cuda.FloatTensor), dumLabel.cuda().type(torch.cuda.FloatTensor)

        target = target.unsqueeze(dim=1)
        output1 = model(data).reshape(batchSize, -1)
        output2 = model(target).reshape(batchSize, -1)
        loss1 = criterion(output1, output2, dumLabel)
        loss = loss1
        testLoss += loss.item() * data.size(0)

    trainLoss = trainLoss / len(trainLoader1.sampler)
    testLoss = testLoss / len(testLoader1.sampler)

    print('Epoch: {} \tTraining Loss: {:.6f} \tTest Loss: {:.6f}'.format(
        epoch, trainLoss, testLoss))

torch.save(model, 'DSAE_{}.pth'.format(epochs))
model.eval()


dt = {}
SAESpecs = []
labels = []
num = 0


trainLoader3 = tqdm.tqdm(trainLoader3)
for data, label in trainLoader3:

    if num > 7000:
        break
    if trainOnGpu:
        data = data.cuda().type(torch.cuda.FloatTensor)

    SAESpec = model(data)
    # print("SAESpec:", SAESpec.shape)
    SAESpec = SAESpec.reshape(SAESpec.shape[0], 1, -1).tolist()
    label = label.tolist()
    SAESpecs.extend(SAESpec)
    labels.extend(label)
    num += 1

dt["labels"] = labels
dt["SAESpecs"] = np.array(SAESpecs)
print("dt_SAESpecs.shape:", dt["SAESpecs"].shape)
trainFile = open("./data/trainData3", "wb")

np.save('./myData/trainData3_data.npy', np.array(SAESpecs), allow_pickle=True)
np.save('./myData/trainData3_label.npy', np.array(labels), allow_pickle=True)

dump(dt, trainFile)
trainFile.close()

dt = {}
SAESpecs = []
labels = []

trainLoader4 = tqdm.tqdm(trainLoader4)

for data, label in trainLoader4:

    if trainOnGpu:
        data = data.cuda().type(torch.cuda.FloatTensor)

    SAESpec = model(data)
    SAESpec = SAESpec.reshape(SAESpec.shape[0], 1, -1).tolist()
    label = label.tolist()
    SAESpecs.extend(SAESpec)
    labels.extend(label)

dt["labels"] = labels
dt["SAESpecs"] = np.array(SAESpecs)
print(dt["SAESpecs"].shape)
trainFile = open("./data/trainData4", "wb")

np.save('./myData/trainData4_data.npy', np.array(SAESpecs), allow_pickle=True)
np.save('./myData/trainData4_label.npy', np.array(labels), allow_pickle=True)

dump(dt, trainFile)
trainFile.close()

dt = {}
SAESpecs = []
labels = []

testLoader3 = tqdm.tqdm(testLoader3)

for data, label in testLoader3:

    if trainOnGpu:
        data = data.cuda().type(torch.cuda.FloatTensor)

    SAESpec = model(data)
    SAESpec = SAESpec.reshape(SAESpec.shape[0], 1, -1).tolist()
    label = label.tolist()
    SAESpecs.extend(SAESpec)
    labels.extend(label)

dt["labels"] = labels
dt["SAESpecs"] = np.array(SAESpecs)
print(dt["SAESpecs"].shape)
trainFile = open("./data/testData3", "wb")

np.save('./myData/testData3_data.npy', np.array(SAESpecs), allow_pickle=True)
np.save('./myData/testData3_label.npy', np.array(labels), allow_pickle=True)

dump(dt, trainFile)
trainFile.close()

dt = {}
SAESpecs = []
labels = []

testLoader4 = tqdm.tqdm(testLoader4)

for data, label in testLoader4:

    if trainOnGpu:
        data = data.cuda().type(torch.cuda.FloatTensor)

    SAESpec = model(data)
    SAESpec = SAESpec.reshape(SAESpec.shape[0], 1, -1).tolist()
    label = label.tolist()
    SAESpecs.extend(SAESpec)
    labels.extend(label)

dt["labels"] = labels
dt["SAESpecs"] = np.array(SAESpecs)
print(dt["SAESpecs"].shape)
trainFile = open("./data/testData4", "wb")

np.save('./myData/testData4_data.npy', np.array(SAESpecs), allow_pickle=True)
np.save('./myData/testData4_label.npy', np.array(labels), allow_pickle=True)

dump(dt, trainFile)
trainFile.close()
