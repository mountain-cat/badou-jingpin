# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 3)
        self.activation = F.softmax
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.linear(x)
        y_pred = self.activation(x, dim=1)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

def build_sample():
    x = np.random.random(5)
    if x[0] >= x[2] and x[4] >= x[2]:
        return x, 0
    elif x[0] <= x[2] and x[4] <= x[2]:
        return x, 2
    else:
        return x, 1

def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

def classify_sample(matrix):
    ret = []
    ret0 = ret1 = ret2 = 0
    for m in matrix:
        if m == 0:
            ret0 += 1
        elif m == 1:
            ret1 += 1
        else:
            ret2 += 1
    ret.append(ret0)
    ret.append(ret1)
    ret.append(ret2)
    return ret


def caculate_max(pred):
    if pred[0] >= pred[1] and pred[0] >= pred[2]:
        return 0
    elif pred[1] >= pred[0] and pred[1] >= pred[2]:
        return 1
    return 2

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    ret = []
    ret = classify_sample(y)
    print("本次预测集中共有%d个0样本，%d个1样本, %d个2样本" % (ret[0], ret[1], ret[2]))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            mmax = -1
            mmax = caculate_max(y_p)
            if mmax == y_t:
                correct += 1
            else:
                wrong += 1

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    print("main")
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    #learning_rate = 0.001  # 学习率
    learning_rate = 0.007  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 画图
    print("xxr--------------------------------------", log)
    plt.plot(range(len(log)), [l[0] for l in log], label="aac")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend() #显示图例
    plt.show() #显示绘制出的图表
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))

    for vec, res in zip(input_vec, result):
        mmax = -1
        mmax = caculate_max(res)
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, mmax, res[mmax]))
        print("-----------------------")


if __name__ == "__main__":
    print("begin----------------")
    main()
    test_vec = [[0.47889086,0.15229675,0.31082123,0.03504317,0.18920843],
                 [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                 [0.78797868,0.67482528,0.13625847,0.34675372,0.99871392],
                 [0.1349776,0.59416669,0.92579291,0.41567412,0.7358894]]
    predict("model.pth", test_vec)

