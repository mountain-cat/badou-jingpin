#-*-coding:utf-8-*-
'''
Author: Shiyao Ma
Date: 2023-09-01 15:11:31
LastEditors: Shiyao Ma
LastEditTime: 2023-09-02 12:03:21
Copyright (c) 2023 by Shiyao Ma, All Rights Reserved. 
'''
# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        # self.index_to_label = {0: '家居', 1: '房产', 2: '股票', 3: '社会', 4: '文化',
        #                        5: '国际', 6: '教育', 7: '军事', 8: '彩票', 9: '旅游',
        #                        10: '体育', 11: '科技', 12: '汽车', 13: '健康',
        #                        14: '娱乐', 15: '财经', 16: '时尚', 17: '游戏'}
        self.index_to_label = {0: 0, 1: 1}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()


    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                tag = line["label"]
                label = self.label_to_index[tag]
                title = line["review"]
                if self.config["model_type"] == "bert":
                    encoded = self.tokenizer(title, max_length=self.config["max_length"], pad_to_max_length=True)
                    input_id = encoded['input_ids']
                    attention_mask = encoded['attention_mask']
                else:
                    input_id, attention_mask = self.encode_sentence(title)
                input_id = torch.LongTensor(input_id)
                attention_mask = torch.LongTensor(attention_mask)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, attention_mask, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id, mask = self.padding(input_id)
        return input_id, mask

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        mask = [1] * len(input_id)
        length2pad = self.config["max_length"] - len(input_id)
        input_id += [0] * length2pad
        mask += [0] * length2pad
        return input_id, mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("valid_tag_news.json", Config)
    print(dg[1])
