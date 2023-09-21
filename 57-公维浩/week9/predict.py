# -*- coding: utf-8 -*-
import torch
import re
import json
import numpy as np
from collections import defaultdict
from config import Config
from model_Bert import TorchModel
from transformers import BertTokenizer
"""
模型效果测试
"""

class SentenceLabel:
    def __init__(self, config, model_path):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.schema = self.load_schema(config["schema_path"])
        self.index_to_sign = dict((y, x) for x, y in self.schema.items())
        self.vocab = self.load_vocab(config["vocab_path"])
        self.model = TorchModel(config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print("模型加载完毕!")

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            schema = json.load(f)
            self.config["class_num"] = len(schema)
        return schema

    # 加载字表或词表
    def load_vocab(self, vocab_path):
        token_dict = {}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1
        self.config["vocab_size"] = len(token_dict)
        return token_dict

    def predict(self, sentence):
        input_ids = self.tokenizer.encode(sentence, max_length=self.config["max_length"], pad_to_max_length=True, add_special_tokens=False)
        target_index = []
        target = []
        with torch.no_grad():
            res = self.model(torch.LongTensor([input_ids]))[0]
            res = torch.argmax(res, dim=-1)
        for start_index, label in enumerate(res):
            end_index = start_index
            for i, j in ((0, 4), (1, 5), (2, 6), (3, 7)):
                if label == i:
                    while res[end_index+1] == j:
                        end_index += 1
                    if end_index != start_index:
                        target_index.append((start_index, end_index))
        for index in target_index:
            target.append(sentence[index[0]: index[1]+1])
        return target


if __name__ == "__main__":
    sl = SentenceLabel(Config, "model_output/epoch_20.pth")
    sentence = "罗伯特从俄罗斯坐高铁去上海博物馆吃烧烤"
    res = sl.predict(sentence)
    print("result:", res)
    # 模型加载完毕!
    # result: ['罗伯特', '俄罗斯', '上海博物馆']
