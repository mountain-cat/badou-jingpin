import torch
import torch.nn as nn
from transformers import BertModel, AutoModelForTokenClassification

def TorchModel(opt):
    return AutoModelForTokenClassification.from_pretrained('badouai/bert', num_labels=opt.n_class)
    
if __name__ == "__main__":
    from config import opt
    net = TorchModel(opt)
    print(net)