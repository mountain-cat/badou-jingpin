import torch
import re
import json
import numpy as np
from collections import defaultdict
from config import Config
from model import TorchModel


"""
模型效果测试
"""

class SentenceLabel:
    def __init__(self, config, model_path):
        self.config = config
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
                token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
        self.config["vocab_size"] = len(token_dict)
        return token_dict

    def predict(self, sentence):
        input_id = []
        for char in sentence:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        with torch.no_grad():
            res = self.model(torch.LongTensor([input_id]))[0]
            # res = torch.LongTensor(res)
           
        labeled_sentence = ""
        for char, label_index in zip(sentence, res):
            labeled_sentence += char + self.index_to_sign[int(label_index)]
        return labeled_sentence

if __name__ == "__main__":
    sl = SentenceLabel(Config, "model_output/epoch_460_microf1_0.658425_macrof1_0.646511.pth")

    sentence = "中国政府决定调整银行的利率"
    res = sl.predict(sentence)
    print(res)

    sentence = "东南亚的气候变得很反常"
    res = sl.predict(sentence)
    print(res)
    
    """
    输出结果：
    模型加载完毕!
    中B-ORGANIZATION国I-ORGANIZATION政I-ORGANIZATION府I-ORGANIZATION决O定O调O整O银O行O的O利O率O
    东B-LOCATION南I-LOCATION亚I-LOCATION的O气O候O变O得O很O反O常O
    """
    
