

import linecache
from torch.utils.data import Dataset, DataLoader
import numpy as np

num_lines = 0
with open('data.txt', 'r', encoding="utf8") as file:
    for line in file:
        num_lines += 1
print("文本文件的行数:", num_lines)


# 文本转化为数字序列，为embedding做准备
def sentence_to_sequence(sentence, vocab):
    sequence = [vocab.get(char, vocab['[UNK]']) for char in sentence]
    return sequence


# 加载字表
def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, "r", encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line.strip()
            vocab[char] = index + 1
    return vocab


# 文本转化为数字序列，为embedding做准备
def sentence_to_sequence(sentence, vocab):
    sequence = [vocab.get(char, vocab['[UNK]']) for char in sentence]
    return sequence


def padding(sequence, max_length):
    if len(sequence) == max_length:
        return np.array(sequence)
    else:
        sequence = sequence[:max_length]
        for i in range(max_length - len(sequence)):
            sequence.append(0)
        return np.array(sequence)


class ReadDataFromFile(Dataset):
    def __init__(self, lines, vocab_path):
        self.data_len = lines
        self.vocab = build_vocab(vocab_path)
        self.max_length = 20

    def __getitem__(self, index):
        # 读取数据，并去除首尾空格
        data_all = linecache.getline('data.txt', index+1) # .strip()
        # 获取训练数据
        data = data_all[2:-1]
        # embedding
        sequence = sentence_to_sequence(data, self.vocab)
        # padding
        x = padding(sequence, self.max_length)
        label = int(data_all[0])
        return x, label

    def __len__(self):
        return self.data_len


def get_data(batch_size=32):
    vocab_path = "vocab.txt"
    train_data = ReadDataFromFile(num_lines, vocab_path)
    train_dataset_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0,
                                      drop_last=True)
    return train_dataset_loader, num_lines/batch_size




































