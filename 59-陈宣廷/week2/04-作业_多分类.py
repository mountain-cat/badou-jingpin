import torch
from torch import nn
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ClassifyTorch(nn.Module):

    def __init__(self, input_size, hidden1_size, hidden2_size, hidden3_size):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, hidden1_size)
        self.layer_2 = nn.Linear(hidden1_size, hidden2_size)
        self.layer_3 = nn.Linear(hidden2_size, hidden3_size)

    def forward(self, input_value):
        y_1 = nn.functional.relu(self.layer_1(input_value))
        y_2 = nn.functional.relu(self.layer_2(y_1))
        y_3 = self.layer_3(y_2)
        return y_3


def get_model(input_size, hidden1_size, hidden2_size, hidden3_size, learn):
    """
    获取模型、损失函数、优化器
    :param input_size: 输入维度
    :param hidden1_size: 第一层隐藏单元个数
    :param hidden2_size: 第二层隐藏单元个数
    :param hidden3_size: 第三层隐藏单元个数
    :param learn: 学习率
    :return: 模型、损失函数、优化器
    """
    Model = ClassifyTorch(input_size, hidden1_size, hidden2_size, hidden3_size)
    loss = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(Model.parameters(), learn)
    return Model, loss, optim


def read_data():
    """
    读取csv数据，用0.8作为训练数据，0.2作为预测数据
    :return:
    """
    num_data = pd.read_csv('数字识别数据集.csv')
    X = num_data.iloc[:, 1:]
    Y = num_data.iloc[:, 0]
    X = torch.tensor(X.values, dtype=torch.float32)
    Y = torch.tensor(Y.values, dtype=torch.long)
    return train_test_split(X, Y, stratify=Y, test_size=0.2)


def acc(y_pre, y):
    y_pre_acc = torch.argmax(y_pre, dim=1)
    return torch.mean((y_pre_acc == y).float())


def train():
    acc_l = []
    loss_l = []
    # 获取模型、损失、优化器
    model, loss, optim = get_model(784, 128, 64, 10, 0.001)
    # 获取数据
    x_train, x_test, y_train, y_test = read_data()
    for i in range(1000):
        loss_epoch_l = []
        for j in range(len(x_train) // 500):
            # 训练数据
            x = x_train[j: (j + 1) * 200]
            y = y_train[j: (j + 1) * 200]
            # 预测
            y_pre = model(x)
            # 计算损失
            loss_step = loss(y_pre, y)
            # 计算梯度
            loss_step.backward()
            # 更新权重
            optim.step()
            loss_epoch_l.append(loss_step.item())
            # 梯度清零
            optim.zero_grad()
        with torch.no_grad():
            # 预测
            y_test_pre = model(x_test)
            # 计算正确率
            accuracy = acc(y_test_pre, y_test)
            # 计算平均loss
            loss_mean = np.mean(loss_epoch_l)
            print(f'epoch：{i+ 1}，测试集上的loss：{loss_mean}，测试集上的正确率：{accuracy}')
            if np.mean(loss_epoch_l) <= 0.001:
                print(f'训练结束，epoch：{i+ 1}，测试集上的loss：{loss_mean}，测试集上的正确率：{accuracy}')
                break
            loss_l.append(np.mean(loss_epoch_l))
            acc_l.append(accuracy)
            loss_epoch_l.clear()
    plt.rcParams['font.sans-serif'] = ['FangSong']
    plt.plot(acc_l, label='正确率')
    plt.plot(loss_l, label='loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train()

