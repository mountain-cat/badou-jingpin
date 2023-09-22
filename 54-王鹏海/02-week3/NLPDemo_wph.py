# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于pytorch的网络编写
用RNN完成一个简单的NLP任务
判断文本总是否有某些特定字符出现
第3周作业
"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(
            len(vocab), vector_dim
        )  # embedding, 将vocab中的每一个离散值转换为向量
        # self.pool = nn.AvgPool1d(sentence_length)  # 池化层, 降低维度，为后续缩减模型大小
        self.singleRNN = nn.RNN(vector_dim,vector_dim, bias=False, batch_first=True)
        self.classify = nn.Linear(vector_dim, 1)  # 线性层
        self.activation = torch.sigmoid #激活函数，使用sigmoid归一化函数
        self.loss = nn.functional.mse_loss #loss函数，采用均方差损失
    
    def forward(self, x, y=None):
        x = self.embedding(x)               #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim), 转换为向量
        # x = x.transpose(1,2)              #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)， 方便后续进行Pool
        # x = self.pool(x)                  #(batch_size, vector_dim, sen_len)->(batch_size, vector_dim, 1), 池化为1个向量
        output, hn = self.singleRNN(x)      #hn: (1, batch_size, vector_dim)
        # print(hn,"hn",hn.shape)
        x = hn.transpose(0,1)               #(1, batch_size, vector_dim) -> (batch_size, 1, vector_dim)
        # print(x,"after hn.transpose", x.shape)
        x = x.squeeze()                     #(batch_size, 1, vector_dim) -> (batch_size, vector_dim), 矩阵的变化
        # print(x,"after x.squeeze", x.shape)
        x = self.classify(x)                #(batch_size, vector_dim) -> (batch_size, 1), 与W进行了相乘，变为一个值
        y_pred  = self.activation(x)        #(batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y) #预测值与真实值计算损失
        else:
            return y_pred   #返回预测值

#构造字符集
#为每一个字符生成一个标号
#{"a":2,"b":3,...}
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index+2
    vocab['unk'] = len(vocab)   # 将未出现在字符集的字符，统一定义为 unk
    vocab['pad'] = len(vocab)
    # print(vocab)
    return vocab

#随机生成一个样本
#从所有字中选取sentence_length个字
def build_sample(vocab, sentence_length):
    #随机从字表中选取，并组成sentence
    x = list(vocab.keys())
    # print(x)
    x = [random.choice(x) for i in range(sentence_length)]
    # print(x)
    #指定哪些字出现时为正样本
    if set("efg") & set(x): #求交集
        y = 1
    else:
        y = 0
    x = [vocab.get(word, vocab['unk']) for word in x] #获取字符对应的键值, 找不到的默认为'unk'
    return x,y

#建立数据集
#输入需要的样本数量
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)     #x本身就是向量
        dataset_y.append([y])   #向量方式保存y
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_sentence_length):
    model.eval()
    x, y = build_dataset(100, vocab, sample_sentence_length)
    print("本次测试集:%d个正样本, %d个负样本" %(sum(y), 100-sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_t, y_p in zip(y, y_pred):
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1
            elif float(y_p) >= 0.5 and int(y_t)==1:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数:%d, 正确率: %f" %(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)

#训练模型
def main():
    #配置模型的参数
    epoch_num = 50      #训练轮数
    batch_size = 20     #每次训练的样本个数
    train_sample = 800  #每轮训练总共的样本数
    char_dim = 20       #每个字的维度
    sentence_length = 8 #样本的文本长度
    learning_rate = 0.001   #学习率

    #建立字表
    vocab = build_vocab()
    #建立模型
    model = build_model(vocab, char_dim,sentence_length)
    #选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log= []
    #开始训练
    for epoch in range(epoch_num):
        model.train()
        watch_loss= []
        for batch in range(int(train_sample/batch_size)):
            x, y = build_dataset(batch_size,vocab,sentence_length)  #构造训练集
            optim.zero_grad()   #梯度归零
            loss = model(x, y)  #计算loss
            loss.backward()     #计算梯度
            optim.step()        #更新权重
            watch_loss.append(loss.item())
            # print(loss)
        print("====\n第%d轮平均Loss:%f" %(epoch+1,np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型
        log.append([acc, np.mean(watch_loss)])
    
    #画图
    plt.plot(range(len(log)),[l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show()

    return model, vocab

#使用模型进行预测
def predict(model, vocab, input_strings):
    char_dim = 20
    sentence_length = 8
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])    #将输入序列化
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, round(float(result[i])), result[i])) #打印结果


if __name__ == "__main__":
    # 训练模型
    model, vocab = main()
    test_strings = ["savghews", "wbsdaarw", "rqwdtghv", "nakfwwql"]
    predict(model, vocab, test_strings)
