#-*-coding:utf-8-*-
'''
Author: Shiyao Ma
Date: 2023-10-27 22:49:34
LastEditors: Shiyao Ma
LastEditTime: 2023-10-28 00:20:44
Copyright (c) 2023 by Shiyao Ma, All Rights Reserved. 
'''
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel, BertConfig

"""
建立网络模型结构
"""

# TorchModel = AutoModelForSequenceClassification.from_pretrained(Config["pretrain_model_path"])

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        max_length = config["max_length"]
        class_num = config["class_num"]
        bert_config = BertConfig('bert-base-chinese')
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], config = bert_config)
        self.classify = nn.Linear(self.bert.config.hidden_size, class_num)
        # self.crf_layer = CRF(class_num, batch_first=True)
        # self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失
        self.config.use_return_dict = True

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        tmp = self.bert(x)
        predict = self.classify(tmp[0]) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)

        if target is not None:
            # if self.use_crf:
            #     mask = target.gt(-1)
            #     return - self.crf_layer(predict, target, mask, reduction="mean")
            # else:
            return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            # if self.use_crf:
            #     return self.crf_layer.decode(predict)
            # else:
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