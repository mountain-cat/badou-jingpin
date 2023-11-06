# coding:utf8

import torch
import torch.nn as nn
import jieba
import numpy as np
import random
import json
from torch.utils.data import DataLoader

"""
基于pytorch的网络编写一个分词模型
我们使用jieba分词的结果作为训练数据
看看是否可以得到一个效果接近的神经网络模型
1、写一个分词器
    读取需要字表，并标号
    读取词表，通过字表把每个词转化成词表向量
    通过jieba，把词表向量及切分结果一一对应组装成训练样本
2、写一个训练模型
    需要考虑输入及输出的变量的形状
    需要把词表向量通过Embedding变成多维向量
    把Embedding之后的的值送入rnn
    把rnn之后的值送不linear,由多维变成2维（因为结果只需要切分或者不切分两种结果）
    用CrossEntropyLoss计算loss
"""


class StrModule(nn.Module):
    def __init__(self, inputs, output, num_rnn_layers, word_list):
        super(StrModule, self).__init__()
        self.embeding = nn.Embedding(len(word_list) + 1, inputs, padding_idx=0)
        self.rnn = nn.RNN(input_size=inputs, hidden_size=output, batch_first=True, num_layers=num_rnn_layers,)
        self.classify = nn.Linear(output, 2)
        self.loss_fun = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, x, y=None):
        x = self.embeding(x)
        x, _ = self.rnn(x)
        y_pred = self.classify(x)
        if y is not None:
            return self.loss_fun(y_pred.view(-1, 2), y.view(-1))
        else:
            return y_pred


# 训练样本构建
# max_text_length：最大文本长度
# word_list：字表
# vocabulary_path：词表地址
class Dataset:
    def __init__(self, max_text_length, word_list, vocabulary_path):
        self.max_text_length = max_text_length
        self.word_list = word_list
        self.vocabulary_path = vocabulary_path
        self.load()

    def load(self):
        self.data = []  # 创建一个变量存储组装好的样本
        with open(self.vocabulary_path, encoding="utf-8") as file:  # with相当于try/chatch,离开with时会自动调用file.close
            for line in file:  # 遍历文件，获取到每一行输入
                # 1、把每行转成数字序列
                # 2、基于jieba，生成分词结果
                # 3、把数字序列中，字数超长的截断，把不够字数的后面补0
                # 4、把jieba分词结果做截断或者补齐动作
                sequence = str_to_sequence(line, self.word_list)
                jieba_result = jieba_result_for_str(line)
                sequence, jieba_result = self.cut_sequence(sequence, jieba_result)
                sequence = torch.LongTensor(sequence)
                jieba_result = torch.LongTensor(jieba_result)
                self.data.append([sequence, jieba_result])
                if len(self.data) > 10000:
                    break

    def cut_sequence(self, sequence, jieba_result):
        sequence = sequence[:self.max_text_length]
        sequence += [0] * (self.max_text_length - len(sequence))
        jieba_result = jieba_result[:self.max_text_length]
        jieba_result += [-100] * (self.max_text_length - len(jieba_result))
        return sequence, jieba_result

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


# 文本转化为数字序列，为embedding做准备
def str_to_sequence(str, vocab):
    str = [vocab.get(char, vocab['unk']) for char in str]
    return str


# 通过jieba获取分词结果
def jieba_result_for_str(line):
    results = jieba.lcut(line)
    jieba_reult = [0] * len(line)  # 初始化一个全都是0，长度为输入文本长度的向量
    pointer = 0  # 初始化一个下标
    for result in results:
        pointer += len(result)
        jieba_reult[pointer - 1] = 1
    return jieba_reult


def build_word_list(word_list_path):
    word_list = {}
    with open(word_list_path, encoding="utf-8") as word:
        for index, line in enumerate(word):
            line = line.strip()
            word_list[line] = index + 1
        word_list['unk'] = len(word_list) + 1
    return word_list


def build_dataset(corpus_path, word_list, max_length, batch_size):
    dataset = Dataset(max_length, word_list, corpus_path)
    data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)  # torch
    return data_loader


# 训练方法
# 1、一轮训练的样本数量
# 2、训练的轮数
# 3、每个字转化成向量的维度
# 4、样本的最大长度
# 5、rnn的层数
# 6、学习率
# 7、隐含层的维度
def main():
    sample_number = 20
    epoch_num = 20
    word_dim = 50
    max_text_length = 20
    rnn_layer = 3
    learning_rate = 1e-3  # 学习率
    output_size = 100
    corpus_path = "../corpus.txt"
    # 先构建字表
    word_list = build_word_list("chars.txt")
    # 构建样本
    data_loder = build_dataset(corpus_path, word_list, max_text_length, sample_number)
    # 创建模型
    model = StrModule(word_dim, output_size, rnn_layer, word_list)
    # 创建优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 开始训练
    for i in range(epoch_num):
        model.train()
        watch_loss = []
        for x, y in data_loder:
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (i + 1, np.mean(watch_loss)))
    # 保存模型
    torch.save(model.state_dict(), "model_str.pth")
    return


# 最终预测
def predict(model_path, vocab_path, input_strings):
    # 配置保持和训练时一致
    char_dim = 50  # 每个字的维度
    hidden_size = 100  # 隐含层维度
    num_rnn_layers = 3  # rnn层数
    vocab = build_word_list(vocab_path)  # 建立字表
    model = StrModule(char_dim, hidden_size, num_rnn_layers, vocab)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的模型权重
    model.eval()
    for input_string in input_strings:
        # 逐条预测
        x = str_to_sequence(input_string, vocab)
        with torch.no_grad():
            result = model.forward(torch.LongTensor([x]))[0]
            result = torch.argmax(result, dim=-1)  # 预测出的01序列
            # 在预测为1的地方切分，将切分后文本打印出来
            for index, p in enumerate(result):
                if p == 1:
                    print(input_string[index], end=" ")
                else:
                    print(input_string[index], end="")
            print()


if __name__ == "__main__":
    main()
    input_strings = ["同时国内有望出台新汽车刺激方案",
                     "沪胶后市有望延续强势",
                     "经过两个交易日的强势调整后",
                     "昨日上海天然橡胶期货价格再度大幅上扬"]
    predict("model_str.pth", "chars.txt", input_strings)
