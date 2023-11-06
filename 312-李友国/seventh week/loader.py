import torch
from torch.utils.data import DataLoader
import csv
from sklearn.model_selection import train_test_split

class DataGenerator:
    def __init__(self, data_path, config,data_class=None):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.data_class = data_class
        self.load()
        self.config["vocab_size"] = len(self.vocab)
        self.config["class_num"] = 2

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] in ['0','1'] and len(row) == 2:
                    lab = row[0]
                    dat = row[1]
                    int_lab = int(lab)
                    input_id = self.encode_sentence(row[1])
                    input_id = torch.LongTensor(input_id)
                    #label_index = torch.LongTensor([int(row[0])])
                    label_index = torch.LongTensor([int_lab])
                    self.data.append([input_id, label_index])

        self.train_set, self.test_set = train_test_split(self.data, test_size=0.2, random_state=42)

    def load_train_set(self):
        return self.train_set

    def load_test_set(self):
        return self.test_set

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        if self.data_class == "train":
            return len(self.train_set)
        elif self.data_class == "test":
            return len(self.test_set)

        return len(self.data)

    def __getitem__(self, index):
        if self.data_class == "train":
            return self.train_set[index]
        elif self.data_class == "test":
            return self.test_set[index]
        return self.data[index]


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True, data_class=None):
    dg = DataGenerator(data_path, config,data_class)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("文本分类练习.csv", Config)
    print(dg[1])