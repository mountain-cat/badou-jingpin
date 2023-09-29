# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])
        self.train_data_size = config["epoch_data_size"]  # 由于采取随机采样，所以需要设定一个采样数量，否则可以一直采
        self.data_type = None  # 用来标识加载的是训练集还是测试集 "train" or "test"
        self.load()

    def load(self):
        self.data = []
        """
        defaultdict(<class 'list'>, {2: [tensor([4270,  157,  164, 1548, 2769, 2685, 3761,  669,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0]), tensor([ 540, 2626,  173,  543,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0]), tensor([ 540, 2626, 2799,  434,  173,  543,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0])]})
        """
        self.knwb = defaultdict(list)
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                # 加载训练集
                if isinstance(line, dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)
                # 加载测试集
                else:
                    self.data_type = "test"
                    assert isinstance(line, list)
                    question, label = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        else:
            assert self.data_type == "test", self.data_type
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == "train":
            return self.random_train_sample()  # 随机生成一个训练样本
        else:
            return self.data[index]

    # 生成cosine_triplet_loss样本
    # s1,s2同类，s3异类
    def random_train_sample(self):
        """
        cosine_triplet_loss样本
        :return: s1,s2同类，s3异类
        """
        standard_question_index = list(self.knwb.keys())
        # 先随机选两类
        p, n = random.sample(standard_question_index, 2)# 从所有标准问题中随机选取两个
        # 如果不足两个问题，则无法选取，所以重新随机一次
        if len(self.knwb[p]) < 2:
            return self.random_train_sample()
        else:
            # 从p里取两个当做s1 和 s2
            s1, s2 = random.sample(self.knwb[p], 2)
            # 从n里取一个当做s3
            s3 = random.choice(self.knwb[n])
        return [s1, s2, s3]
        # 随机正样本
        # if random.random() <= self.config["positive_sample_rate"]:
        #     p = random.choice(standard_question_index)
        #     # 如果选取到的标准问下不足两个问题，则无法选取，所以重新随机一次
        #     if len(self.knwb[p]) < 2:
        #         return self.random_train_sample()
        #     else:
        #         s1, s2 = random.sample(self.knwb[p], 2)  # 在同一标准问题下随机选取两个问题
        #         return [s1, s2, torch.LongTensor([1])]  # 返回的正样本
        # # 随机负样本
        # else:
        #     p, n = random.sample(standard_question_index, 2)  # 从所有标准问题中随机选取两个
        #     # 再从这两个标准问题中分别随机各选取一个问题，就构成了负样本
        #     s1 = random.choice(self.knwb[p])
        #     s2 = random.choice(self.knwb[n])
        #     return [s1, s2, torch.LongTensor([-1])]  # 返回的负样本。 正样本标签1 负样本标签-1，是配合loss来设置的，模型用的是CosineEmbeddingLoss


# 加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


# 加载schema
def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator("valid_tag_news.json", Config)
    print(dg[1])
