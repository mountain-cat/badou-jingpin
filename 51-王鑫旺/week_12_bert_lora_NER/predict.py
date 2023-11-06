import torch
import re
import json
import numpy as np
from collections import defaultdict
from config import Config
from model import loramodel
from peft import get_peft_model, LoraConfig, TaskType

"""
模型效果测试
"""

class SentenceLabel:
    def __init__(self, config, model_path):
        self.config = config
        self.schema = self.load_schema(config["schema_path"])
        self.index_to_sign = dict((y, x) for x, y in self.schema.items())
        self.vocab = self.load_vocab(config["vocab_path"])
        #######################################################
        ###################修改部分############################
        #######################################################
        self.model = loramodel
        peft_config = LoraConfig(
                task_type=TaskType.TOKEN_CLS,
                inference_mode=True,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["query", "key", "value"]
            )
        self.model = get_peft_model(self.model, peft_config)
        # print(self.model.state_dict().keys())
        state_dict = self.model.state_dict()
        state_dict.update(torch.load(model_path))
        # print("==========================================================")
        # print(state_dict)
        self.model.load_state_dict(state_dict)
        # print(self.model.state_dict().keys())
        
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
        res = res.squeeze()
        
        for char, label_index in zip(sentence, res):
            print(int(torch.argmax(label_index)))
            print(label_index)
            labeled_sentence += char + self.index_to_sign[int(torch.argmax(label_index))]
        return labeled_sentence

if __name__ == "__main__":
    sl = SentenceLabel(Config, "model_output/epoch_41_microf1_0.630600_macrof1_0.630058.pth")

    sentence = "邓小平中国政府南非朱利亚斯东南大学"
    res = sl.predict(sentence)
    print(res)

    sentence = "东南亚的气候变得很反常"
    res = sl.predict(sentence)
    print(res)

