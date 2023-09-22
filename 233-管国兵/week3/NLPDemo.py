# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch的网路编写
实现一个网路完成一个简单的nlp任务
判断文本中是否有某些字符出现
加入RNN层

"""

class TorchModel(nn.Module):
    def __init__(self, embedding_dim, sentence_len, vocab):
        """
        构建模型变量（用到哪些层）
        :param embedding_dim: embedding的向量维度
        :param sentence_len: 文本长度
        :param vocab: 词表字典
        """
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embedding_dim, padding_idx=0)  # embedding 输出维度 (batch_size,sentence_len,embedding_dim)

        # (batch_size,sentence_len,embedding_dim) -> (batch_size,embedding_dim,embedding_dim)
        self.rnn = nn.RNN(num_layers=1, input_size=embedding_dim, hidden_size=embedding_dim, batch_first=True)
        self.classify = nn.Linear(embedding_dim, 1)  # 线性层
        self.activation = torch.sigmoid  # sigmoid归一化激活函数
        self.loss = nn.functional.mse_loss  # 均方差loss

    def forward(self, x, y=None):
        """
        计算流程，定义计算流程
        :param x: 输入参数，一个batch的样本
        :param y: 真实y
        :return: 预测值 或 loss
        """
        # (batch_size,sentence_len)的序号 -> (batch_size,sentence_len, vector_dim)的tensor
        embedding_out = self.embedding(x)
        _, x = self.rnn(embedding_out)
        x = self.classify(x.squeeze())  # (batch_size, 1)
        y_pred = self.activation(x)  # (batch_size, 1)
        if y is not None:
            loss = self.loss(y_pred, y)
            return loss
        else:
            return y_pred


def build_vocab():
    """
    构建词表 \n
    每个字符对应一个序号，unk为最后一个，pad为第一个
    :return: 字典词表
    """
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab = {'pad': 0}
    for index, char in enumerate(chars):
        # 为每个字符对应一个序号
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_len):
    """
    随机生成一个样本对 \n
    含 x y z为正样本。不会存在词表外字符
    :param vocab: 词表
    :param sentence_len: 字符串长度
    :return: 样本对，如 [19, 8, 25, 11, 27] , 1
    """
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_len)]  # 随机选 sentence_len个字符
    # 出现了xyz中任意一个，设置y为1，为正样本
    if set("xyz") & set(x):
        y = 1
    else:
        y = 0
    # 转为序号返回，便于做embedding
    # print(x) # ['s', 'h', 'y', 'k', 'unk']
    x = [vocab.get(char, vocab['unk']) for char in x]
    # [19, 8, 25, 11, 27] , 1
    return x, y


def build_dataset(sample_len, vocab, sentence_len):
    """
    构建样本对 \n
    构建samplex)
    loss = model(xml个样本对
    :param sample_len: 需要的样本个数
    :param vocab: 词表
    :param sentence_len: 字符串长度
    :return: tensor_dataset_x:(sample_len, sentence_len)，tensor_dataset_y:(sample_len, 1)
    """
    dataset_x = []
    dataset_y = []
    for i in range(sample_len):
        x, y = build_sample(vocab, sentence_len)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)


def build_model(char_dim, sentence_len, vocab):
    """
    建立模型
    :param char_dim: embedding时每个字符的向量的长度，会被随机初始化为权重，参与训练
    :param sentence_len: 字符串的长度
    :param vocab: 词表字典
    :return: model
    """
    model = TorchModel(char_dim, sentence_len, vocab)
    return model


def evaluate(model, vocab, sample_len):
    """
    测试模型正确率,为了在训练时每轮输出正确率 \n
    构建一些样本，在每轮跑完的时候，使用模型预测这些样本，监控模型每轮的正确率 \n
    1、构建测试集 \n
    2、输出测试集信息：测试集大小，正样本个数，负样本个数
    :param model: 模型
    :param vocab: 词表字典
    :param sample_len:样本长度 同 sentence_len
    :return: 正确率
    """
    model.eval()  # 测试模式
    x, y = build_dataset(200, vocab, sample_len)
    print("本次预测集中有%d个正样本，%d个负样本" % (sum(y), 200 - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_pred, y_true in zip(y_pred, y):
            if float(y_pred) < 0.5 and int(y_true) == 0:
                correct += 1
            elif float(y_pred) >= 0.5 and int(y_true) == 1:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    epoch_num = 20  # 训练轮数
    learning_rate = 0.005  # 学习率
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮总样本数
    char_dim = 20  # 每个字符的维度
    sentence_len = 6  # 样本字符串长度
    # 建立词表
    vocab = build_vocab()
    # 建立模型
    model = build_model(char_dim, sentence_len, vocab)
    # 优化器，更新权重
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练
    for epoch in range(epoch_num):
        model.train()  # 训练模式
        watch_loss = []  # 记录每批次loss
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_len)
            loss = model(x, y)  # 算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("======\n第%d轮，平均loss %f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_len)  # 测试本轮正确率
        log.append([acc, np.mean(watch_loss)])
    # 输出embedding weight
    # print('embedding.weight \n', model.embedding.weight)
    # 画图
    plt.plot(range(len(log)), [a[0] for a in log], label='acc')  # 画正确率
    plt.plot(range(len(log)), [l[1] for l in log], label='loss')  # 画loss
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), 'model.pth')
    # 保存词表
    writer = open('vocab.json', 'w', encoding='utf8')
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


def predict(model_path, vocab_path, input_strings):
    """
    接收字符串列表，预测类别
    :param model_path: 模型路径
    :param vocab_path: 词典路径
    :param input_strings: 预测样本
    :return:
    """
    print('\n======预测====== \n含x 或 y 或z 为正样本，类别为1，否则类别为0')
    char_dim = 20
    sentence_len = 6
    vocab = json.load(open(vocab_path, 'r', encoding='utf8'))
    model = build_model(char_dim, sentence_len, vocab)
    model.load_state_dict(torch.load(model_path))
    x = []
    for input_string in input_strings:
        x.append([vocab[s] for s in input_string]) # 转为序号 (20,6,4)
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))  # 预测
    for i, input_string in enumerate(input_strings):
        print("输入: %s, 预测类别：%d, 概率值：%f" % (input_string, round(float(result[i])), result[i]))


if __name__ == "__main__":
    main()

    test_strings = ['xyzjid', 'njdefs', 'asderf', 'yhjkiu']
    predict('model.pth', 'vocab.json', test_strings)
