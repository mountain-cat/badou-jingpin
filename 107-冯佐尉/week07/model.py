import torch.nn as nn
from transformers import BertModel


class BertModule(nn.Module):

    def __init__(self,config):
        super(BertModule,self).__init__()
        self.__config = config
        self.__bert = BertModel.from_pretrained("../bert", return_dict=False).to(self.__config["device"])
        self.__lstm = nn.LSTM(self.__config["input_dim"], self.__config["input_dim"], num_layers=self.__config["lstm_layers"], batch_first=True).to(self.__config["device"])
        self.__linear = nn.Linear(self.__config["input_dim"], self.__config["output_dim"]).to(self.__config["device"])
        self.__loss = nn.CrossEntropyLoss()

    def forward(self,x,y=None):
        _,x = self.__bert(x)
        x,_ = self.__lstm(x)
        y_pred = self.__linear(x)

        if y is not None:
            return self.__loss(y_pred, y.squeeze())
        else:
            return y_pred