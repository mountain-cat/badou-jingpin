# -*- coding: utf-8 -*-

import json
import re
from collections import defaultdict

import torch
from transformers import BertTokenizer

from config import Config
from model import TorchModel

"""
NER模型预测
"""


class NERLabel:
    def __init__(self, config, model_path):
        self.config = config
        self.schema = self.load_schema(config["schema_path"])
        self.vocab = self.load_vocab(config["vocab_path"])
        self.max_length = config["max_length"]
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"], add_special_tokens=False)
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
                token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
        self.config["vocab_size"] = len(token_dict)
        return token_dict

    def predict(self, sentence):
        input_id = self.bert_encode_sentence(sentence)
        with torch.no_grad():
            res = self.model(torch.LongTensor([input_id]))[0]
            if not self.config["use_crf"]:
                res = torch.argmax(res, dim=-1)
        if not self.config["use_crf"]:
            pred_label = res.cpu().detach().tolist()
        pred_entities = self.decode(sentence, pred_label)
        return pred_entities

    def bert_encode_sentence(self, text):
        return self.tokenizer.encode(text, padding='max_length', max_length=self.max_length, truncation=True)

    def decode(self, sentence, label):
        label = "".join([str(x) for x in label[:len(sentence)]])
        results = defaultdict(list)
        for location in re.finditer("(04+)", label):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", label):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", label):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", label):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return results


if __name__ == "__main__":
    ner = NERLabel(Config, "model_output/epoch_10.pth")

    sentence = "湖南省省长杨正午代表说:要使湖南的农业发生质的转变,就要以市场为导向,科技为先导,抓好一个基础,做好三篇文章。"
    res = ner.predict(sentence)
    print(res)
