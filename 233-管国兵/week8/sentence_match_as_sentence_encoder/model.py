# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

"""
建立网络模型结构
"""


class SentenceEncoder(nn.Module):
    """
    输入一个文本序列，输出一个文本序列的表示向量
        由后面的一个模型（SiameseNetwork）把训练的部分补充起来
    """
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    # 输入为问题字符编码
    def forward(self, x):
        # sentence_length = torch.sum(x.gt(0), dim=-1)
        x = self.embedding(x)
        # 使用lstm
        # x, _ = self.layer(x)
        # 使用线性层
        x = self.layer(x)
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x  # 输出的是纯向量，pooling后的向量


class SiameseNetwork(nn.Module):
    """
    孪生网络
        需要一个编码器（本身是一个模型），建议单独写一个类SentenceEncoder
        还需要一个loss
    """

    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)  # 编码器
        # self.loss = nn.CosineEmbeddingLoss()  # 正向为1，负向为-1

    # 计算余弦距离  1-cos(a,b)
    # cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine

    def cosine_triplet_loss(self, a, p, n, margin=None):
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        if margin is None:
            diff = ap - an + 0.1
        else:
            diff = ap - an + margin.squeeze()
        return torch.mean(diff[diff.gt(0)]) # gt(0)就是大于0的值，gt就是greater than

    # 前项计算
    # sentence : (batch_size, max_length)
    def forward(self, sentence1, sentence2=None, sentence3=None):
        # 同时传入两个句子
        if sentence2 is not None:
            # 共享权重写法：两个sentence都经过同一个encoder模型
            vector1 = self.sentence_encoder(sentence1)  # vec:(batch_size, hidden_size) 文本1得到向量1
            vector2 = self.sentence_encoder(sentence2)  # 文本2得到向量2
            vector3 = self.sentence_encoder(sentence3)  # 文本2得到向量2
            # 如果有标签，则计算loss
            if sentence2 is not None and sentence3 is not None:  # 如果传入了sentence1 sentence2 sentence3，就是模型在训练，应该计算loss
                return self.cosine_triplet_loss(vector1, vector2, vector3)
            else:  # 如果只传入了sentence1 sentence2，则认为模型在测试，计算余弦距离
                return self.cosine_distance(vector1, vector2)
        # 单独传入一个句子时，认为正在使用向量化能力
        else:
            return self.sentence_encoder(sentence1)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config

    Config["vocab_size"] = 10
    Config["max_length"] = 4
    model = SiameseNetwork(Config)
    s1 = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    s2 = torch.LongTensor([[1, 2, 3, 4], [3, 2, 3, 4]])
    l = torch.LongTensor([[1], [0]])
    y = model(s1, s2, l)
    print(y)
    # print(model.state_dict())
