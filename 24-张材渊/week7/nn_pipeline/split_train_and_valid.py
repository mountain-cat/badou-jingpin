import torch
import csv
import json
from torch.utils.data import Dataset, DataLoader
from nn_pipeline.config import Config


class MyDataSet(Dataset):
    def __init__(self, loaded_data):
        self.data = loaded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def split_raw_data():
    data = []
    with open(Config['raw_data_path'], 'r', encoding='UTF-8') as f:
        cr = csv.reader(f)
        for row in cr:
            data.append(row)
        data = data[1:]

    custom_dataset = MyDataSet(data)
    print('原始数据数量',len(custom_dataset))
    train_size = int(len(custom_dataset) * 0.7)
    test_size = len(custom_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, test_size])
    print("训练集数量","测试集数量",len(train_dataset),len(test_dataset))
    # 保存样本为文件
    file_train = open(Config["train_data_path"], 'w', encoding='UTF-8')
    for d in train_dataset:
        s = {"tag": d[0], "title": d[1], "content": ''}
        file_train.write(json.dumps(s, ensure_ascii=False) + '\n')
    file_train.close()
    file_valid = open(Config["valid_data_path"], 'w', encoding='UTF-8')
    for d in test_dataset:
        s = {"tag": d[0], "title": d[1], "content": ''}
        file_valid.write(json.dumps(s, ensure_ascii=False) + '\n')
    file_valid.close()

    # 统计正样本数量和负样本数量以及文本平均长度
    positive_count = 0
    negative_count = 0
    all_len = 0
    for d in train_dataset:
        assert d[0] in ["0", "1"]
        if d[0] == "1":
            positive_count += 1
        else:
            negative_count += 1
        all_len += len(d[1])
    print("正样本数量", positive_count)
    print("负样本数量", negative_count)
    print("平均文本长度", all_len / len(train_dataset))
    return positive_count, negative_count, all_len / len(test_dataset)
