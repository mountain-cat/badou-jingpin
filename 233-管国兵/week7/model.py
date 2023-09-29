# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel
from loader import DataLoader
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, input_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        # 原始代码
        self.embedding = nn.Embedding(len(vocab) + 1, input_dim)
        self.layer = nn.Linear(input_dim, input_dim)
        self.pool = nn.MaxPool1d(sentence_length)

        self.classify = nn.Linear(input_dim, 2)
        self.activation = torch.sigmoid     #sigmoid做激活函数
        self.dropout = nn.Dropout(0.5)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # 原始代码
        x = self.embedding(x)  #input shape:(batch_size, sen_len) (10,6)
        x = self.layer(x)      #input shape:(batch_size, sen_len, input_dim) (10,6,20)
        x = self.dropout(x)    #input shape:(batch_size, sen_len, input_dim)
        x = self.activation(x) #input shape:(batch_size, sen_len, input_dim)
        x = self.pool(x.transpose(1,2)).squeeze() #input shape:(batch_size, sen_len, input_dim)


        x = self.classify(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y.squeeze())
        else:
            return y_pred