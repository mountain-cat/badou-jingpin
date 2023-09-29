import torch
import torch.nn as nn
from transformers import BertModel

class TorchModel(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()

        self.encoder = BertModel.from_pretrained('badouai/bert', return_dict=False)
        self.cls = nn.Linear(opt.hidden_size, opt.n_class)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, y=None):
        x, _ = self.encoder(x)      # 取bert的整个句子的每一个字符的输出维度为(batch_size, seq_length, hidden_size)
        y_pred = self.cls(x)
        if y is not None:
            loss = self.loss_func(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
            return loss
        return y_pred
    
if __name__ == "__main__":
    from config import opt
    net = TorchModel(opt)
    print(net)