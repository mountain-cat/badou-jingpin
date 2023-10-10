import torch
from transformers import BertTokenizer
from transformers.utils import PaddingStrategy

from config import Config
from model import BertModule
from loader import read_schema


class NerPredict:

    def __init__(self,config,model_path):
        self.__config = config
        self.__model = BertModule(self.__config)
        self.__model.load_state_dict(torch.load(model_path))
        self.__model.eval()
        self.__tokenizer = BertTokenizer.from_pretrained("../bert")
        print("模型加载完毕!")

    def __sentence_embbeding(self,sentence):
        content = self.__tokenizer.encode_plus(sentence, padding=PaddingStrategy.MAX_LENGTH,
                                              max_length=self.__config["sentence_max_length"])
        content = torch.LongTensor(content["input_ids"]).to(self.__config["device"])
        return content

    def predict(self,sentence):
        if len(sentence) > self.__config["sentence_max_length"]:
            sentence = sentence[:self.__config["sentence_max_length"]-1]

        input_ids = self.__sentence_embbeding(sentence)
        input_ids = input_ids.unsqueeze(0)
        y_pred = self.__model(input_ids)
        y_pred = y_pred.squeeze(0)
        y_pred = torch.argmax(y_pred, dim=-1)

        return self.__result_process(sentence,y_pred)

    def __result_process(self,sentence,y_pred):
        labels = y_pred.cpu().detach().tolist()
        schema = read_schema(self.__config)
        schema = dict((value, key) for key, value in schema.items())
        result = []
        for index,char in enumerate(sentence):
            result.append((char, schema[labels[index+1]]))

        return result

if __name__ == "__main__":
    ner = NerPredict(Config,"model.pth")

    res = ner.predict("中国的首都在北京，是世界上最大的城市")

    print(res)