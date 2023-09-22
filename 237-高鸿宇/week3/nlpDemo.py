#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现
"""

class RNNModel(nn.Module):
    '''
        定义RNN基类
        args:
            rnn_layer(nn.Module): rnn层,可能为rnn, 也可能为lstm等等
            vocab_size(int): 词表大小
            vector_dim(int): 需要embedding的词向量的维度
    '''
    def __init__(self, rnn_layer, vocab_size, vector_dim) -> None:
        super().__init__()
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        self.embedding = nn.Embedding(vocab_size, vector_dim)
        self.cls = nn.Linear(self.num_hiddens, self.vocab_size)
        self.num_directions = 1
    
    def forward(self, x, state):
        x = self.embedding(x)
        # (batch_size, seq_length, vector_dim) -> (seq_length, batch_size, vector_dim)
        x = x.permute(1, 0, 2)
        y, state = self.rnn(x, state)
        # 由于nn.RNN的输出为(seq_length, batch_size, num_hiddens), 因此对seq_length这个维度取平均即可
        y = self.cls(y.mean(dim=0).reshape(-1, y.shape[-1]))
        return y, state
    
    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return  torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device))

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  #字符集
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index   #每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab

#随机生成一个样本
#从所有字中选取sentence_length个字
#反之为负样本
def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    #指定哪些字出现时为正样本
    if set("abc") & set(x):
        y = 1
    #指定字都未出现，则为负样本
    else:
        y = 0
    # dict().get(), 根据输入的key去寻找相应的value值, 若key不存在则根据设置好的default参数返回相应值
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.tensor(dataset_y, dtype=torch.uint8)


#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    print("本次预测集中共有%d个正样本, %d个负样本"%(sum(y), 200 - sum(y)))
    with torch.no_grad():
        state = model.begin_state(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), batch_size=x.shape[0])
        y_pred, _ = model(x, state)      #模型预测
        y_hat = nn.Softmax(dim=1)(y_pred)
        y_hat = torch.argmax(y_hat, dim=1)
        correct = (y_hat == y).sum()
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (200)))
    return correct / 200

def main():
    #配置参数
    epoch_num = 20        #训练轮数
    batch_size = 16       #每次训练样本个数
    train_sample = 512    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 6   #样本文本长度
    learning_rate = 0.005 #学习率
    num_hidden = 64       #RNN隐藏层的隐藏层节点数
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loss = nn.CrossEntropyLoss()
    # 建立字表
    vocab_dict = build_vocab()
    # 建立模型
    rnn_layer = nn.RNN(char_dim, num_hidden)
    net = RNNModel(rnn_layer, len(vocab_dict), char_dim)
    # 选择优化器
    optim = torch.optim.Adam(net.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        net.train()
        watch_loss = []
        state = None
        for _ in range(int(train_sample / batch_size)):
            state = net.begin_state(batch_size=batch_size, device=device)
            x, y = build_dataset(batch_size, vocab_dict, sentence_length) #构造一组训练样本
            optim.zero_grad()    #梯度归零
            y_hat, _ = net(x, state)
            l = loss(y_hat, y)
            l.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(l.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(net, vocab_dict, sentence_length)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    #画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show()
    #保存模型
    torch.save(net.state_dict(), "week3/model.pth")
    # 保存词表
    writer = open("week3/vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab_dict, ensure_ascii=False, indent=2))
    writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings, num_hidden):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab_dict = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    rnn_layer = nn.RNN(char_dim, num_hidden)
    net = RNNModel(rnn_layer, len(vocab_dict), char_dim)
    net.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab_dict[char] for char in input_string])  #将输入序列化
    net.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        state = net.begin_state(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), batch_size=len(x))
        result, _ = net(torch.LongTensor(x), state)
        pred = nn.Softmax(dim=1)(result)
    for input_string, res in zip(input_strings, pred):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, torch.max(res, 0)[1], torch.max(res, 0)[0])) #打印结果


if __name__ == "__main__":
    main()
    test_strings = ["favfee", "wbsdfg", "rqwdeg", "nakwww"]
    predict("week3/model.pth", "week3/vocab.json", test_strings, 64)
