# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from TorchCRF import CRF
from transformers import BertModel

"""
建立网络模型结构
"""


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        class_num = config["class_num"]
        """
        bert在前，crf在后
        bert的输出，传给crf的输入
        """
        self.bert = BertModel.from_pretrained(config["pre_train_model_path"], return_dict=False)
        # self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=1)
        self.classify = nn.Linear(hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)  # 无batch_first参数
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  # loss采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        # x = self.embedding(x)  # input shape:(batch_size, sen_len)
        # x, x2 = self.layer(x)  # input shape:(batch_size, sen_len, input_dim)
        x, pooler_output = self.bert(x)
        predict = self.classify(x)
        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)  # 128,50 [[True,False.....], [],[]]
                return - self.crf_layer(predict, target, mask, reduction="mean") # predict(128,50,4),target(128,50) [[0,0,0,...-1,-1,-1]],mask(128,50)
            else:
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.viterbi_decode(predict)
            else:
                return predict


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config

    model = TorchModel(Config)
