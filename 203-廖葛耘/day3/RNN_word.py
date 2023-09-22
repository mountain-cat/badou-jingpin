# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  # embedding层
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)  # RNN层
        self.classify = nn.Linear(vector_dim, 1)  # 线性层
        self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = nn.functional.mse_loss  # loss函数采用均方差损失

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        rnn_output, _ = self.rnn(x)  # RNN处理 (batch_size, sen_len, vector_dim) -> (batch_size, sen_len, vector_dim)
        x = rnn_output[:, -1, :]  # 取最后一个时刻的输出 (batch_size, sen_len, vector_dim) -> (batch_size, vector_dim)
        x = self.classify(x)  # (batch_size, vector_dim) -> (batch_size, 1)
        y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index  # 每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    y = realOutput(x)

    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, y


def realOutput(x):
    # 指定哪些字出现时为正样本
    if set("ed") & set(x):
        y = 1
        # 指定字都未出现，则为负样本
    else:
        y = 0
    return y


def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    learning_rate = 0.005  # 学习率
    # 建立字表
    vocab = build_vocab()
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 建立200个用于测试的样本
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), 200 - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1  # 负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1  # 正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测

    correct = 0
    wrong = 0
    for i, input_string in enumerate(input_strings):
        realVal = realOutput([char for char in input_string])
        predVal = round(float(result[i]))
        if realVal == predVal:
            correct += 1
        else:
            wrong += 1
        print("输入：%s, 预测类别：%d, 真实值：%d, 概率值：%f" % (input_string, predVal, realVal, result[i]))  # 打印结果
    print('总共输入：%d, 正确数量：%d ，错误数量：%d' % (len(input_strings), correct, wrong))

if __name__ == "__main__":
    main()
    test_strings = ["hsxuorehdl", "erzbkbnvcc", "sfsxkztzar", "jqphmefwrv", "zzxyhcwbkc",
                    "xfakexhecd", "cdlsesvnmb", "ikvvuiskcr", "otiowmigry", "slwhdoudfg", "mleienanwx",
                    "fmrdlmknlw", "klhibrbmiv", "ovhimstuhs", "apkvqjjrqs", "revtrjnodm", "rxqbhdvgyt",
                    "tsdxarqlcr", "cobltvonvw", "hrjojrvwoc"]
    predict("model.pth", "vocab.json", test_strings)
