import json
import math
import random
from collections import defaultdict

import torch
from torch.utils import data
from transformers import BertTokenizer
from transformers.utils import PaddingStrategy

from config import Config


class BaseDataset(data.Dataset):

    def __init__(self, config,type_dict):
        super(BaseDataset, self).__init__()
        self._config = config
        self._tokenizer = BertTokenizer.from_pretrained("../bert")

        self.__type_dict = type_dict
        self.__dataset = self._load_data(type_dict)

    def _load_data(self, type_dict):
        raise NotImplementedError

    def encoding(self, sentence):
        content = self._tokenizer.encode_plus(sentence, padding=PaddingStrategy.MAX_LENGTH,
                                               max_length=self._config["senence_max_length"])
        content = torch.LongTensor(content["input_ids"]).to(self._config["device"])
        return content

    def get_type_dict(self):
        return self.__type_dict

    def __getitem__(self, index):
        return *self.__dataset[index],

    def __len__(self):
        return len(self.__dataset)


class TrainDataset(BaseDataset):

    def __init__(self,config,type_dict):
        super(TrainDataset, self).__init__(config,type_dict)

    def _load_data(self,type_dict):
        """
        组装数据
        :param type_dict: {key:value}
        :return:
            输出数据格式: [(sentence,positive_sentence,negative_sentence),...]
        """
        datas = self.__read_file(type_dict)
        result = []
        for key in datas:
            for i in range(1,len(datas[key])):
                result.append((self.encoding(datas[key][i]),
                               self.encoding(self.__get_positive_sentence(datas, key)),
                               self.encoding(self.__get_negative_sentence(datas, key))))
        return result

    def __get_positive_sentence(self,questions_dict,key):
        return random.sample(questions_dict[key], 1)[0]

    def __get_negative_sentence(self,questions_dict,key):
        index = list(questions_dict.keys())[:]
        index.remove(key)
        new_index = random.sample(index, 1)[0]
        value = random.sample(questions_dict[new_index],1)[0]
        return value

    def __read_file(self, type_dict):
        """
        读取数据文件
        :param type_dict: {key:value}
            数据文件格式： questions:[],target:""
        :return:
            输出数据格式: {target:[question,...],...}
        """
        datas = defaultdict(list)
        with open(self._config["train_file"], encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                questions = line["questions"]
                target = type_dict[line["target"]]

                datas[target].extend(questions)

        return datas


class ValidDataset(BaseDataset):

    def __init__(self,config,type_dict):
        super(ValidDataset, self).__init__(config,type_dict)

    def _load_data(self, type_dict):
        result = []
        with open(self._config["valid_file"], encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                result.append((self.encoding(line[0]),
                               type_dict[line[1]]))
        return result


def read_schema(config):
    schema = {}
    with open(config["schema_file"], "r", encoding="utf8") as f:
        schema = json.load(f)
    return schema


def load_data(config):
    type_dict = read_schema(config)
    train_dataset = TrainDataset(config,type_dict)
    valid_dataset = ValidDataset(config,type_dict)
    return train_dataset,valid_dataset


if __name__ == '__main__':
    load_data(Config)