import json
import math
import random
from collections import defaultdict

import torch
from torch.utils import data
from transformers import BertTokenizer
from transformers.utils import PaddingStrategy

from config import Config

class NerDataset(data.Dataset):

    def __init__(self, config,data_file,type_dict):
        super(NerDataset, self).__init__()
        self._config = config
        self._tokenizer = BertTokenizer.from_pretrained("../bert")

        self.__data_file = data_file
        self.__type_dict = type_dict
        self.__dataset = self._load_data(type_dict)

    def _load_data(self, type_dict):
        """
        char1 type
        ...

        charn type

        :param type_dict: {key:value}
        输出格式[[sentence1,...],[label1,...]]
        :return:
        """
        datas = []
        with open(self.__data_file, encoding="utf8") as f:
            sentences = f.read().split("\n\n")
            for sentence in sentences:
                sentence_output = []
                label_output = torch.full([self._config["sentence_max_length"]],-1).to(self._config["device"])
                lines = sentence.split("\n")
                for index,line in enumerate(lines):
                    if index+3 > self._config["sentence_max_length"]:
                        break

                    if len(line) > 0:
                        char,label = line.split(" ")
                        sentence_output.append(char)
                        label_output[index+1] = type_dict[label]

                sentence_def = "".join(sentence_output)
                # print(f"sentence:{sentence_str},len:{len(sentence_str)}")
                sentence_str = self.encoding(sentence_def)
                # print(f"encoding:{sentence_str},len:{len(sentence_str)}")

                datas.append((sentence_str,label_output,sentence_def))

        return datas

    def encoding(self, sentence):
        content = self._tokenizer.encode_plus(sentence, padding=PaddingStrategy.MAX_LENGTH,
                                               max_length=self._config["sentence_max_length"])
        content = torch.LongTensor(content["input_ids"]).to(self._config["device"])
        return content

    def get_type_dict(self):
        return self.__type_dict

    def __getitem__(self, index):
        return *self.__dataset[index],

    def __len__(self):
        return len(self.__dataset)


def read_schema(config):
    schema = {}
    with open(config["schema_file"], "r", encoding="utf8") as f:
        schema = json.load(f)
    return schema


def load_data(config):
    type_dict = read_schema(config)
    train_dataset = NerDataset(config,config["train_file"],type_dict)
    valid_dataset = NerDataset(config,config["valid_file"],type_dict)
    return train_dataset,valid_dataset


if __name__ == '__main__':
    load_data(Config)