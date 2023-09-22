# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import torch.optim as optim

"""

基于pytorch框架编写模型训练
实现一个自行构造的找分类(机器学习)任务
分类：x是一个6维向量，
    如果第1个数>第2个数，则为第0类
    如果第3个数>第4个数，则为第1类
    如果第4个数>第5个数，则为第2类

"""

#定义2层线性层的模型，每层后都接激活层
class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(TorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1) # 6 , 9
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size1, hidden_size2) # 9 , 3
        self.loss = nn.CrossEntropyLoss # 交叉熵

    def forward(self, x, y=None):
        hidden = self.layer1(x) # 20*6 -> 20*9
        y_pred = self.activation(hidden)
        y_pred = self.layer2(y_pred) # 20 * 3
        y_pred = self.activation(y_pred)
        if y is not None:
            loss_fn = nn.CrossEntropyLoss()  # 创建交叉熵损失函数实例
            loss = loss_fn(y_pred, torch.argmax(y, dim=1))  # 使用交叉熵损失函数计算损失
            return loss
        else:
            return y_pred

# 分类规则，创建训练样本
# x是长度6的向量，y是类别
def build_sample():
    x = np.random.random(6)
    if x[0] > x[1]:
        return x, 0 # 第一类
    elif x[2] > x[3]:
        return x, 1# 第二类
    elif x[4] > x[5]:
        return x, 2# 第三类
    else:
        return build_sample();

def to_one_hot(target, shape=3):
    one_hot_target = np.zeros(shape)

    one_hot_target[target] = 1
    return one_hot_target

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    while len(X) < total_sample_num:
        x, y = build_sample()
        X.append(x)
        Y.append(to_one_hot(y))
    return torch.FloatTensor(X), torch.FloatTensor(Y)

def main():
    epoch_num = 100 # 训练轮数
    batch_size = 20 # 每次训练样本个数
    train_sample = 5000 # 每轮训练总样本个数
    input_size = 6 # 输入向量的维度
    hidden_size1 = 9 # 隐含层维度
    hidden_size2 = 3 # 最后输出层维度
    learning_rate = 0.001 # 学习率
    # 建立模型
    model = TorchModel(input_size, hidden_size1, hidden_size2)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample) # 5000 对训练样本

    for epoch in range(epoch_num): # 一轮一轮来训练，这里是训练10轮
        model.train() # 模型调为训练模式
        watch_loss = [] # 记录loss
        for batch_index in range(train_sample // batch_size): # 循环 5000/20 次，依次取20对样本进行训练
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward() # 计算梯度
            optim.step() # 更新权重
            optim.zero_grad() # 梯度归零
            watch_loss.append(loss.item()) # loss汇总
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    return

if __name__ == "__main__":
    # main()

    input_size = 6
    model = TorchModel(input_size, 3, 3)
    model.load_state_dict(torch.load("model.pth"))  # 加载训练好的权重
    model.eval()  # 测试模式
    test_vec = [[0.47889086,0.15229675,0.31082123,0.03504317,0.18920843,0.18920843],
                [0.44963533,0.5524256,0.95758807,0.90520434,0.04890681,0.48920843],
                [0.58797868,0.67482528,0.13625847,0.34675372,0.99871392,0.18920843],
                [0.99349776,0.59416669,0.32579291,0.41567412,0.5358894,0.98920843]]
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(test_vec))  # 模型预测
        '''
            如果第1个数>第2个数，则为第0类
            如果第3个数>第4个数，则为第1类
            如果第4个数>第5个数，则为第2类
        '''
    for vec, res in zip(test_vec, result): # 结果
        print('测试样本：', vec)
        print("预测类别：", torch.argmax(res).item())
        print('\n')
