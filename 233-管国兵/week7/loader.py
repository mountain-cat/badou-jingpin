import csv
import random
from config import Config
import torch

"""
加载训练数据
"""


class DataLoader(object):
    def __init__(self, data_path, vocab_path, dataset_size, text):
        self.data_path = data_path
        self.data = load_data(data_path)
        self.vocab_path = vocab_path
        self.vocab = self.load_vocab()
        self.vocab_size = len(self.vocab)
        self.dataset_size = dataset_size
        self.text = text

    def load_vocab(self):
        return load_data(self.data_path)

    def get_data(self):
        return self.data

    def get_data_size(self):
        return len(self.data)

    def get_random_data(self, n):
        return random_choice_data(self.data, n)

    def get_dataset(self):
        return build_dataset(self.data_path, self.vocab_path, self.dataset_size)

    def get_encode_sentence(self):
        token_dict = load_vocab(self.vocab_path)
        return encode_sentence(self.text, token_dict)

def load_data(train_data_path):
    data = []
    with open(train_data_path, "r", encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # 跳过第一行标题
        for row in csv_reader:
            data_row = []
            data_row.append(int(row[0]))
            data_row.append(row[1])
            data.append(data_row)
    return data


def random_choice_data(data, num):
    random_data = []
    for i in range(num):
        random_data.append(random.choice(data))
    return random_data


def load_vocab(vocab_path):
    """
    加载词表
    {'[UNK]': 1, '[SPACE]': 2, '!': 3, '"': 4, '#': 5, '$': 6, '%': 7, '&': 8 ... }
    :param vocab_path:
    :return:
    """
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


# 补齐或截断输入的序列，使其可以在一个batch内运算
def padding(input_id):
    """
    padding or truncating input_id
    补齐或者截取input_id，使其长度为max_length
    :param input_id:
    :return:
    """
    input_id = input_id[:Config["max_length"]]
    input_id += [0] * (Config["max_length"] - len(input_id))
    return input_id


def encode_sentence(text, token_dict):
    """
    将输入的文本转为id(vocab的序号)
    特别好吃，量特大，而且送餐特别快，特别特别棒 -> 234 12 34 56 78 90 10
    :param text:
    :param token_dict:
    :return:
    """
    input_id = []
    for char in text:
        input_id.append(token_dict.get(char, token_dict["[UNK]"]))
    input_id = padding(input_id)
    return input_id


def build_dataset(train_data_path, vocab_path, dataset_size):
    token_dict = load_vocab(vocab_path)
    data = load_data(train_data_path)
    random_data = random_choice_data(data, dataset_size)
    dataset_x = []
    dataset_y = []
    for label, text in random_data:
        dataset_x.append(encode_sentence(text, token_dict))
        dataset_y.append([label])
    dataset_x = torch.LongTensor(dataset_x)
    return dataset_x, torch.FloatTensor(dataset_y)

# dataset_x, dataset_y = build_dataset(Config['train_data_path'], Config['vocab_path'], 100)
# print(dataset_x, dataset_y)


# token_dict = load_vocab(Config['vocab_path'])
# data = load_data(Config['train_data_path'])
# random_data = random_choice_data(data, 10)
# # print(random_data)
# for i in random_data:
#     input_id = encode_sentence(i[1], token_dict)
#     print(input_id)


# data = load_data(Config['train_data_path'])
# print(data)

# random_data = random_choice_data(data, 10)
# print(random_data)

# token_dict = load_vocab(Config['vocab_path'])
# print(token_dict)
