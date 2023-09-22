#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Xie Dexiao
# datetime： 2023/7/26 下午8:54 
# ide： PyCharm


import numpy as np
import torch
import torch.nn as nn
import random
import json
from torch import optim
from torch.utils.data import Dataset, DataLoader


class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)  # 线性层
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.activation = torch.sigmoid

    def forward(self, x):
        x = x.type(self.linear1.weight.dtype)
        out = self.linear1(x)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out)
        out = self.activation(out)
        return out


def train_data_gen(sampleNum, featureNum, labelNum):
    # 生成随机的特征数据
    features = np.random.randn(sampleNum, featureNum)
    # 生成随机的类别标签
    labels = np.random.randint(0, labelNum, size=sampleNum)
    return features, labels


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train():
    # 定义训练数据集
    sampleNum = 5000
    featureNum = 100
    labelNum = 5
    features, labels = train_data_gen(sampleNum, featureNum, labelNum)
    train_dataset = CustomDataset(features, labels)

    # 定义训练数据加载器
    batch_size = 10
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 定义模型和优化器
    input_dim = featureNum  # 输入特征的维度
    hidden_dim = 16  # 隐藏层的维度
    output_dim = 5  # 输出类别的数量
    model = TorchModel(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        loss = 0
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")


    # 测试数据
    testData = torch.Tensor(np.random.randn(10, 100))

    predictions = model(testData)
    _, predicted_labels = torch.max(predictions, 1)

    print(predicted_labels)


if __name__ == '__main__':
    train()
