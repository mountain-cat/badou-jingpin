from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import jieba
import numpy as np
from transformers import BertModel


class BertSegmentation(nn.Module):

    def __init__(self,input_dim):
        super(BertSegmentation, self).__init__()
        self.bert = BertModel.from_pretrained(r"../bert", return_dict=False).to(get_device())
        self.classify = nn.Linear(input_dim, 2).to(get_device())
        self.loss = nn.CrossEntropyLoss(ignore_index=-100).to(get_device())

    def forward(self, x, y=None):
        x,_ = self.bert(x)
        y_pred = self.classify(x)
        if y is not None:
            # view(-1,2): (batch_size, sen_len, 2) ->  (batch_size * sen_len, 2)
            y_pred_tmp = y_pred.view(-1, 2)
            y_tmp = y.view(-1)
            return self.loss(y_pred_tmp, y_tmp)
        else:
            return y_pred


class Dataset:
    def __init__(self, corpus_path, vocab, max_length):
        self.vocab = vocab
        self.corpus_path = corpus_path
        self.max_length = max_length
        self.load()

    def load(self):
        self.data = []
        with open(self.corpus_path, encoding="utf8") as f:
            for line in f:
                sequence = sentence_to_sequence(line, self.vocab)
                label = sequence_to_label(line)
                sequence, label = self.padding(sequence, label)
                sequence = torch.LongTensor(sequence).to(get_device())
                label = torch.LongTensor(label).to(get_device())
                self.data.append([sequence, label])
                #使用部分数据做展示，使用全部数据训练时间会相应变长
                if len(self.data) > 10000:
                    break

    #将文本截断或补齐到固定长度
    def padding(self, sequence, label):
        sequence = sequence[:self.max_length]
        sequence += [0] * (self.max_length - len(sequence))
        label = label[:self.max_length]
        label += [-100] * (self.max_length - len(label))
        return sequence, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


#文本转化为数字序列，为embedding做准备
def sentence_to_sequence(sentence, vocab):
    sequence = [vocab.get(char, vocab['[UNK]']) for char in sentence]
    return sequence


#基于结巴生成分级结果的标注
def sequence_to_label(sentence):
    words = jieba.lcut(sentence)
    label = [0] * len(sentence)
    pointer = 0
    for word in words:
        pointer += len(word)
        label[pointer - 1] = 1
    return label


#加载字表
def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, "r", encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line.strip()
            vocab[char] = index + 1   #每个字对应一个序号
    return vocab


#建立数据集
def build_dataset(corpus_path, vocab, max_length, batch_size):
    dataset = Dataset(corpus_path, vocab, max_length) #diy __len__ __getitem__
    data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size) #torch
    return data_loader


def train():
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    char_dim = 768  # 每个字的维度
    max_length = 20  # 样本最大长度
    learning_rate = 1e-3  # 学习率
    vocab_path = "../bert/vocab.txt"  # 字表文件路径
    corpus_path = "corpus.txt"  # 语料文件路径
    vocab = build_vocab(vocab_path)  # 建立字表
    data_loader = build_dataset(corpus_path, vocab, max_length, batch_size)  # 建立数据集
    model = BertSegmentation(char_dim)  # 建立模型
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 建立优化器
    # 训练开始
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, y in data_loader:
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    return


#最终预测
def predict(model_path, vocab_path, input_strings):
    #配置保持和训练时一致
    char_dim = 768  # 每个字的维度
    vocab = build_vocab(vocab_path)       #建立字表
    model = BertSegmentation(char_dim)   #建立模型
    model.load_state_dict(torch.load(model_path))   #加载训练好的模型权重
    model.eval()
    for input_string in input_strings:
        #逐条预测
        x = sentence_to_sequence(input_string, vocab)
        with torch.no_grad():
            result = model.forward(torch.LongTensor([x]).to(get_device()))[0]
            result = torch.argmax(result, dim=-1).to(get_device())  #预测出的01序列
            #在预测为1的地方切分，将切分后文本打印出来
            for index, p in enumerate(result):
                if p == 1:
                    print(input_string[index], end=" ")
                else:
                    print(input_string[index], end="")
            print()


if __name__ == '__main__':
    # train()
    input_strings = ["同时国内有望出台新汽车刺激方案",
                     "沪胶后市有望延续强势",
                     "经过两个交易日的强势调整后",
                     "昨日上海天然橡胶期货价格再度大幅上扬"]

    predict("model.pth", "../bert/vocab.txt", input_strings)