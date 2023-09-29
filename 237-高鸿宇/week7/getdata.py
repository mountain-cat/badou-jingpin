import pandas as pd
import torch
from transformers import BertTokenizer
from config import opt

class Dataset():
    def __init__(self, opt, is_train:bool=True) -> None:
        self.data = pd.read_csv(opt.filepath)
        self.is_train = is_train
        self.opt = opt
        self.tokenizer = BertTokenizer.from_pretrained('bert')
        self.load()

    def load(self):
        # 统计每一种类别的数量, 保证类别数量均衡
        self.positive_data = self.data[self.data['label'] == 1]
        self.negtive_data = self.data[self.data['label'] == 0].sample(4000)
        n_train = int(0.8 * len(self.negtive_data))
        if self.is_train:
            self.dataset = pd.concat([self.positive_data.iloc[:n_train], self.negtive_data.iloc[:n_train]])
        else:
            self.dataset = pd.concat([self.positive_data.iloc[n_train:], self.negtive_data.iloc[n_train:]])
    
    def __getitem__(self, i):
        label = self.dataset.iloc[i]['label']
        comment = self.dataset.iloc[i]['review']
        sequence = sentence_to_sequence(self.tokenizer, comment)
        sequence = self.padding(sequence)
        sequence, label = torch.LongTensor(sequence), torch.LongTensor([label])
        return sequence, label
    
    def __len__(self):
        return len(self.dataset)

    def padding(self, sequence):
        '''
        根据max_length对句子进行截取和padding
        '''
        sequence = sequence[:self.opt.max_length]
        sequences_with_padding = sequence+[0]*(self.opt.max_length - len(sequence))
        return sequences_with_padding


def sentence_to_sequence(tokenizer, sentence:str):
    '''
    将语句转换为序列, 为后续embedding使用

    args:
        sentence(str): 需要转换的句子
    '''
    sequence = tokenizer.encode(sentence)
    return sequence[:-1]


if __name__ == "__main__":
    data = Dataset(opt)
    data.__getitem__(5066)