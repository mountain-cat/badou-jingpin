import torch.nn as nn
import torch

import copy
from transformers import BertModel,BertConfig


class orther(nn.Module):
    def __init__(self):
            super(orther, self).__init__()
            self.c=nn.Sequential(
                nn.Linear(768,768),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(768,256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256,256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256,4)
            )
            config = BertConfig.from_pretrained("D:\deeplean\china_split/bert-base-chinese\config.json")
            self.bert = BertModel(config)
            # self.bert.load_state_dict(torch.load("D:\deeplean\china_split/bert-base-chinese/pytorch_model.bin"), strict=False)
    def forward(self, x):#x输入为（bitch,256,768）
             x=self.bert(x)[0]

             return self.c(x)