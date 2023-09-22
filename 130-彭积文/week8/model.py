import torch.nn as nn
from transformers import BertModel


class BertModule(nn.Module):

    def __init__(self,config):
        super(BertModule,self).__init__()
        self.__config = config
        self.__encoder = BertEncoder(self.__config)
        self.__loss = nn.TripletMarginLoss(margin=self.__config["loss_margin"]).to(self.__config["device"])

    def forward(self,sentence,positive_sentence=None,negative_sentence=None):
        sentence_embedding = self.__encoder(sentence)

        if positive_sentence is not None:
            positive_sentence_embedding = self.__encoder(positive_sentence)

            if negative_sentence is not None:
                negative_sentence_embedding = self.__encoder(negative_sentence)

                return self.__loss(sentence_embedding,positive_sentence_embedding,negative_sentence_embedding)
            else:
                return sentence_embedding,positive_sentence_embedding
        else:
            return sentence_embedding


class BertEncoder(nn.Module):

    def __init__(self,config):
        super(BertEncoder,self).__init__()
        self.__config = config
        self.__bert = BertModel.from_pretrained("../bert", return_dict=False).to(self.__config["device"])
        self.__linear = nn.Linear(self.__config["hidden_size"],self.__config["hidden_size"]).to(self.__config["device"])

    def forward(self,sentence):
        _, sentence_embedding = self.__bert(sentence)
        sentence_embedding = self.__linear(sentence_embedding)

        return sentence_embedding