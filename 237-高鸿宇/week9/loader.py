import json
import torch
from config import opt
from torch.utils.data import DataLoader
from transformers import BertTokenizer


class Dataset:
    def __init__(self, opt, istrain=True) -> None:
        self.opt = opt
        self.data_path = opt.train_data_path if istrain else opt.valid_data_path
        self.schema = self.load_schema(opt.schema_path)
        self.tokenizer = BertTokenizer.from_pretrained('badouai/bert')
        self.sentences = []
        self.load()
    
    def load(self):
        self.data = []
        with open(self.data_path, encoding='utf-8') as f:
            # 文本文件结构是每一行只有一个字和相应的label，每一句话结尾都有一个\n\n换行符
            segments = f.read().split('\n\n')
            # 读取到的内容为一个列表，列表中的每一个元素为文本文件中的一个句子
            for segment in segments:
                sentence = []
                labels = []
                # segment为完整的一句话, 内容为'char label\n char label\n.....'
                # 因此以\n作为分隔读取每一个字符以及对应的label
                for line in segment.split('\n'):
                    if line.strip() == '':
                        continue
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])
                self.sentences.append(''.join(sentence))
                input_ids = self.encode_sentence(sentence)
                labels = self.padding(labels, -1)
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return
    
    def encode_sentence(self, sentence, padding=True):
        # bert的tokenizer的编码器会在句子头尾自动加入[cls]与[sep]因此去掉这两个token
        input_ids = self.tokenizer.encode(sentence)[1:-1]
        if padding:
            input_ids = self.padding(input_ids)
        return input_ids
    
    def padding(self, input_ids, pad_token=0):
        input_ids = input_ids[:self.opt.max_length]
        input_ids += [pad_token] * (self.opt.max_length - len(input_ids))
        return input_ids

    def load_schema(self, file_path):
        with open(file_path, encoding='utf-8') as f:
            return json.load(f)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]

def load_data(opt, shuffle=True):
    dataset = Dataset(opt, istrain=shuffle)
    data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=shuffle)
    return data_loader

if __name__ == "__main__":
    Dataset(opt)