# -*- coding: utf-8 -*-

from transformers import BertTokenizer


class BertPretrain:
    def __init__(self, config):
        self.max_length = config["max_length"]
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])

    def bert_encode_sentence(self, text):
        return self.tokenizer.encode(text, padding='max_length', max_length=self.max_length, truncation=True)
