import torch.nn as nn
from transformers import BertModel


class BertModule(nn.Module):

    def __init__(self,config):
        super(BertModule,self).__init__()
        self.__config = config
        self.__encoder = BertEncoder(self.__config)
        self.__loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self,x,y=None):
        y_pred = self.__encoder(x)

        if y is not None:
            y_tmp = y.view(-1)
            y_pred_tmp = y_pred.view(-1,y_pred.shape[-1])
            return self.__loss(y_pred_tmp,y_tmp)
        else:
            return y_pred


class BertEncoder(nn.Module):

    def __init__(self,config):
        super(BertEncoder,self).__init__()
        self.__config = config
        self.__bert = BertModel.from_pretrained("../bert", return_dict=False).to(self.__config["device"])
        self.__linear = nn.Linear(self.__config["hidden_size"],self.__config["type_num"]).to(self.__config["device"])

    def forward(self,sentence):
        sentence_embedding,_ = self.__bert(sentence)
        sentence_embedding = self.__linear(sentence_embedding)

        return sentence_embedding