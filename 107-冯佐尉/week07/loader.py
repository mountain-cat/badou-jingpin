import math
from collections import defaultdict

import torch
from torch.utils import data
from transformers import BertTokenizer
from transformers.utils import PaddingStrategy

from config import Config


class CsvDataset(data.Dataset):

    def __init__(self,config,datas):
        super(CsvDataset, self).__init__()
        self.__config = config
        self.__tokenizer = BertTokenizer.from_pretrained("../bert")

        self.__read_csv(datas)

    def __read_csv(self,datas):
        self.__dataset = []

        for data in datas:
            content = self.__tokenizer.encode_plus(data[1], padding=PaddingStrategy.MAX_LENGTH,max_length=self.__config["senence_max_length"])
            content = torch.LongTensor(content["input_ids"]).to(self.__config["device"])
            label = torch.LongTensor([int(data[0])]).to(self.__config["device"])

            self.__dataset.append([content,label])

    def __getitem__(self, index):
        return self.__dataset[index][0],self.__dataset[index][1]

    def __len__(self):
        return len(self.__dataset)


def read_data_from_file(config):
    """
    :param config:
    :return: {"0":[], "1":[]}
    """
    datas = defaultdict(list)
    # read csv
    is_first_line = True
    with open(config["file_path"], encoding="utf8") as f:
        for line in f:
            if is_first_line:
                is_first_line = False
                continue

            cols = line.strip().split(",")
            datas[cols[0]].append(cols[1])

    return datas


def read_data(config,rate=0.1):
    """
    :param config:
    :param rate: 分割比例 0.1-0.9
    :return: train_data,valid_data
    """
    datas = read_data_from_file(config)

    train_data = [[0,data] for data in datas["0"][:math.ceil(len(datas["0"])*(1-rate))]]
    valid_data = [[0,data] for data in datas["0"][math.ceil(len(datas["0"])*(1-rate)):]]

    train_data.extend([[1,data] for data in datas["1"][:math.ceil(len(datas["1"])*(1-rate))]])
    valid_data.extend([[1,data] for data in datas["1"][math.ceil(len(datas["1"])*(1-rate)):]])

    return CsvDataset(config,train_data),CsvDataset(config,valid_data)


if __name__ == '__main__':
    read_data(Config)