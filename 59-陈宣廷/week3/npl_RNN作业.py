import random
import string

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def create_vocab():
    """
    创建词汇表，包含所有英文字母（大小写）
    :return: 词汇表
    """
    vocab_dict = {}
    for ind, vocab in enumerate(string.ascii_letters):
        vocab_dict[vocab] = ind
    vocab_dict['unk'] = len(vocab_dict)
    return vocab_dict


def create_train_data():
    """
    创建1000条训练数据
    :return: 训练数据
    """
    vocab_dict = create_vocab()
    x_list = []
    y_list = []
    for i in range(1000):
        s = ''.join(random.sample(string.ascii_letters, 7))
        if set('bcd') & set(s):
            y_list.append(0)
        else:
            y_list.append(float(1))
        x_list.append([vocab_dict[j] for j in s])
    return torch.tensor(x_list), torch.tensor(y_list)


def acc(y_pre, y):
    """
    计算正确率
    :return: 正确率
    """
    Y_pre = y_pre.squeeze()
    Y_pre_numpy = Y_pre.numpy()
    Y_pre = (Y_pre_numpy >= 0.5).astype(np.float)
    accuracy = np.mean((Y_pre == y.numpy()).astype(np.float32))
    return accuracy


class NlpRnn(nn.Module):
    def __init__(self, vocab_len, vector_len, vocab_count):
        """
        基于RNN的NLP任务
        :param vocab_len: 字符集长度
        :param vector_len: 每个字母用多长的数字表示
        :param vocab_count: 每个字符串包含的字母个数
        """
        super(NlpRnn, self).__init__()
        self.embedding = nn.Embedding(vocab_len, vector_len)  # embedding层
        self.pooling = nn.AvgPool1d(vocab_count)  # 池化层
        self.rnn = nn.RNN(vector_len, 1)
        self.activation = torch.tanh
        self.loss = F.mse_loss

    def forward(self, x, y=None):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.pooling(x)
        x = x.squeeze()
        x = self.rnn(x)
        out = self.activation(x[0])
        if y is not None:
            return self.loss(out.squeeze(), y)
        else:
            return out


epoch = 1000  # 训练轮式
batch_size = 20  # 训练批次
lr = 0.001  # 学习率
X, Y = create_train_data()
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, stratify=Y)


def train():
    model = NlpRnn(53, 10, 7)
    optim = torch.optim.Adam(model.parameters(), lr)
    acc_list = []
    loss_list = []
    for i in range(epoch):
        loss_l = []
        for j in range(len(train_x) // batch_size):
            x = train_x[j * batch_size: (j + 1) * batch_size]
            y = train_y[j * batch_size: (j + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度清零
            loss_l.append(loss.item())
        with torch.no_grad():
            y_pre = model(test_x)
            accuracy = acc(y_pre, test_y)
            acc_list.append(accuracy)
            loss_list.append(np.mean(loss_l))
            print(f'在测试数据上，epoch：{i}，loss：{np.mean(loss_l)}，acc：{accuracy}')
            if accuracy == 1:
                break
    plt.plot(acc_list, label='acc')
    plt.plot(loss_list, label='loss')
    plt.scatter(len(acc_list) - 1, acc_list[-1])
    plt.scatter(len(acc_list) - 1, loss_list[-1])
    plt.text(len(acc_list) - 1, acc_list[-1], acc_list[-1])
    plt.text(len(acc_list) - 1, loss_list[-1], loss_list[-1])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train()

