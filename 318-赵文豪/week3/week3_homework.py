#  尝试在nlpdemo中使用rnn模型训练。

import torch
import torch.nn as nn
import numpy as np
import json
import random
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self, dim_vector, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), dim_vector)
        self.rnn = nn.RNN(input_size=dim_vector, hidden_size=dim_vector, batch_first=True,num_layers=1)
        self.linear = nn.Linear(dim_vector, 5)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.embedding(x)
        _, x = self.rnn(x)
        y_pre = self.linear(x.squeeze())
        if y is not None:
            return self.loss(y_pre, y)
        else:
            return y_pre

def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index
    vocab["unk"] = len(vocab)
    return vocab


# 生成一个样本，长度为len_sentence，5分类
# 含“abc"：[0, 1, 0, 0, 0]
# 含"lmn"：[0, 0, 1, 0, 0]
# 含"opq"：[0, 0, 0, 1, 0]
# 含"uvw"：[0, 0, 0, 0, 1]
# 都不含：[1, 0, 0, 0, 0]
def build_sample(vocab, len_sentence):
    x = [random.choice(list(vocab.keys())) for _ in range(len_sentence)]
    if set(x) & set("abc"):
        y = [0, 1, 0, 0, 0]
    elif set(x) & set("lmn"):
        y = [0, 0, 1, 0, 0]
    elif set(x) & set("opq"):
        y = [0, 0, 0, 1, 0]
    elif set(x) & set("uvw"):
        y = [0, 0, 0, 0, 1]
    else:
        y = [1, 0, 0, 0, 0]
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, y


def build_data(vocab, len_sentence, num):
    data_x = []
    data_y = []
    for i in range(num):
        x, y = build_sample(vocab, len_sentence)
        data_x.append(x)
        data_y.append(y)
    return torch.LongTensor(data_x), torch.FloatTensor(data_y)


def model_evaluate(model, vocab, len_sentence, num):
    model.eval()
    x, y = build_data(vocab, len_sentence, num)
    correct = 0
    with torch.no_grad():
        # for x, y in enumerate(data):
        #     y_pre = model(x)
        #     if torch.argmax(y_pre) == np.argmax(y):
        #         correct += 1
        y_pre = model(x)
        y_pre = torch.argmax(y_pre, dim=-1)  # -1:最后一个维度
        y = torch.argmax(y, dim=-1)  # -1:最后一个维度
        correct += int(sum(y == y_pre))
    return correct / num


def model_train():
    epoch = 20
    batch_side = 20
    learning_rate = 0.005
    data_num = 10000
    dim_vector = 20
    len_sentence = 10
    vocab = build_vocab()
    model = TorchModel(dim_vector, vocab)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    plot_data = []
    for i in range(epoch):
        model.train()
        epoch_loss = []
        for batch in range(int(data_num / batch_side)):
            x, y = build_data(vocab, len_sentence, batch_side)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            epoch_loss.append(loss.item())
        acc = model_evaluate(model, vocab, len_sentence, 100)
        plot_data.append([acc, np.mean(epoch_loss)])
        print("=========\n第%d轮平均loss:%f" % (i + 1, np.mean(epoch_loss)))
    plt.plot(range(len(plot_data)), [p[0] for p in plot_data], label="acc")
    plt.plot(range(len(plot_data)), [p[1] for p in plot_data], label="loss")
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 使用训练好的模型做预测
def model_predict(model_path, vocab_path, input_strings):
    dim_vector = 20  # 每个字的维度
    len_sentence = 10  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = TorchModel(dim_vector, vocab)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char, vocab["unk"]) for char in input_string])  # 将输入序列化
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d, 概率值：%s" % (input_string, int(torch.argmax(result[i])), result[i]))  # 打印结果


if __name__ == "__main__":
    model_train()
    test_strings = ["yevx4efske", "hrsdfggkdl", "rqaeqgadbh", "nlazwwwhld", "diuhsfv4ow"]  # 4 2 1 1 3
    model_predict("model.pth", "vocab.json", test_strings)
