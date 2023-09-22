import numpy as np
import torch
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel
import os

UNCASED = './bert-base-chinese' 
VOCAB = 'vocab.txt' 
tokenizer=BertTokenizer.from_pretrained(r"D:\deeplean\china_split\bert-base-chinese", return_dict=False, max_length=256)
bert = BertModel.from_pretrained(r"D:\deeplean\china_split\bert-base-chinese", return_dict=False)
sen_code = tokenizer(text="我真的很烦诶", return_tensors='pt', padding='max_length', truncation=True, max_length=256)
t=bert(**sen_code)
print(t[0])