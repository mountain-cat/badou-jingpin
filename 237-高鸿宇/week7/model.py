import torch
from torch import nn
from transformers import BertModel

class NlpModel(nn.Module):
    def __init__(self, hidden_dim:int, n_classes:int=2) -> None:
        '''
        文本分类模型初始化函数

        args:
            hidden_dim(int): 隐藏层维度
            n_classes(int): 最终的预测类别数量
        '''
        super().__init__()

        self.decoder = BertModel.from_pretrained('bert', return_dict=False)
        self.cls = nn.Linear(hidden_dim, n_classes)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x:torch.Tensor, y:torch.Tensor=None):
        _, x = self.decoder(x)   # 取bert的pooler output, 输出的是整个句子的分类头, 维度为(batch_size, hidden_dim)
        y_pred = self.cls(x)
        if y is not None:
            loss = self.loss_func(y_pred, y.view(-1))
            return loss
        else:
            return y_pred
        
def get_net(hidden_dim:int, n_classes:int, weight_to_load:str=None):
    net = NlpModel(hidden_dim, n_classes)

    if weight_to_load is not None:
        checkpoint = torch.load(weight_to_load)
        net.load_state_dict(checkpoint)

    return net

if __name__ == "__main__":
    net = NlpModel(768)
    y = net(torch.LongTensor([[1, 2, 3, 4, 5],
                              [2, 3, 4, 5, 6]]))
    print(y.shape)
    y = net(torch.LongTensor([[1, 2, 3, 4, 5],
                              [2, 3, 4, 5, 6]]), torch.LongTensor([1, 0]))
    print(y)