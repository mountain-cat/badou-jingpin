# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的三分类(机器学习)任务
规律：x是一个0.001到0.999的数，依据x所在的区间进行分类

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 3)  # 线性层
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(3, 3)
        self.loss = nn.CrossEntropyLoss()# 多分类问题适合使用交叉熵
    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):#此方法为重写的向前传播函数，在调用model(x,y)时实际上是调用了此方法
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        x = self.activation(x)
        y_pred = self.linear2(x)
        
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律

def build_sample():
    x = np.random.randint(1,999)
    if x >=600:
        return [x/1000], [1,0,0]
    if x <=300:
        return [x/1000], [0,1,0]
    else:
        return [x/1000], [0,0,1]

# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个一类，%d个二类,%d个三类" % (sum([row[0] for row in y]), sum([row[1] for row in y]),sum([row[2] for row in y])))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            i_p = y_p.tolist().index(max(y_p))
            i_t = y_t.tolist().index(max(y_t))
            if (i_p == i_t):
                correct += 1  # 样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 1000  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 1000  # 每轮训练总共训练的样本总数
    input_size = 1  # 输入向量维度
    learning_rate = 0.02  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):#进行epoch_num轮次的训练
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):# [//]代表整数除法，返回值为整数
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 1
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%d" % (vec, res.tolist().index(max(res)),max(res)))  # 打印结果


if __name__ == "__main__":
    main()
    #test_vec = [[0.3],[0.05], [0.7],[0.5]]
    #predict("model.pth", test_vec)
