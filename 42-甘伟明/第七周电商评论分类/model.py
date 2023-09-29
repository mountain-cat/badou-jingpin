

import torch.nn as nn
from transformers import BertModel
import torch
import torch.nn.functional as f


softmax = nn.Softmax(dim=-1)


def custom_loss(y_actual, y_pred):  # y_actual是标签值，y_pred是模型的预测值
    y_pred = y_pred.view(-1)
    # 成本敏感权重
    bias_0 = 1.15
    bias_1 = 1.0
    sigmoid_cross_entropy = -(y_actual * torch.log(y_pred + 1e-12) * bias_1 + (1 - y_actual) * torch.log(1 - y_pred + 1e-12) * bias_0)
    return sigmoid_cross_entropy.mean()


class Bert(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(r".\BERT", return_dict=False)
        self.classify = nn.Linear(768, 1)

    def forward(self, x, y=None):
        sequence_output, pooler_output = self.bert(x)
        y_pred = self.classify(pooler_output)
        y_pred = f.sigmoid(y_pred)

        if y is not None:
            return y_pred, custom_loss(y, y_pred)
        else:
            return y_pred