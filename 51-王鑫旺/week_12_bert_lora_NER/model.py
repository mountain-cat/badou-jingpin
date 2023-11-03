# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel, AutoModelForTokenClassification

from config import Config

"""
建立网络模型结构
"""
#######################################################
###################修改部分############################
#######################################################
loramodel = AutoModelForTokenClassification.from_pretrained(r"D:\desktop\NLP\bert_chinese",
                                                            num_labels = Config["class_num"])

##########################################################

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


