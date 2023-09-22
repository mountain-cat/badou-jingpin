# -*- coding: utf-8 -*-
import torch
import re
import json
import numpy as np
from collections import defaultdict
from config import Config
from model import TorchModel
from transformers import BertTokenizer
import jieba

"""
模型效果测试
"""


class SentenceLabel:
    def __init__(self, config, model_path):
        self.config = config
        self.vocab = self.load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = self.load_schema(config["schema_path"])
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.model = TorchModel(config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print("模型加载完毕!")

    def load(self, sentenece):
        if self.config["model_type"] == "bert":
            input_ids = self.tokenizer.encode(sentenece, max_length=self.config["max_length"],
                                              pad_to_max_length=True, add_special_tokens=False)
        else:
            input_ids = self.encode_sentence(sentenece)
        return torch.LongTensor(input_ids).reshape(1, -1)

    def encode_sentence(self, text, padding=True):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:
            input_id = self.padding(input_id)
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

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
                token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
        self.config["vocab_size"] = len(token_dict)
        return token_dict

    def predict(self, sentence):
        input_id = self.load(sentence)
        with torch.no_grad():
            pred_results = self.model(input_id)[0]
            if not self.config["use_crf"]:
                pred_results = torch.argmax(pred_results, dim=-1)
                pred_label = pred_results.cpu().detach().tolist()
            pred_entities = self.decode(sentence, pred_label)
        return pred_entities

    def decode(self, sentence, labels):
        labels = "".join([str(x) for x in labels[:len(sentence)]])
        results = defaultdict(list)
        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return results


if __name__ == "__main__":
    sl = SentenceLabel(Config, "model_output/epoch_20.pth")

    sentence = "在北京家里的客厅的颜色比较稳重但不沉重相反很好的表现了欧式的感觉给人高雅的味道"
    res = sl.predict(sentence)
    print(res)

    sentence = "双子座的健康运势也呈上升的趋势但下半月有所回落"
    res = sl.predict(sentence)
    print(res)
