# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务




规律：x是一个4维向量，按x中第二大的数的位置分类，分为4类。



"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 20)  # 线性层
        self.layer2 = nn.Linear(20, 4)
        self.activation1 = torch.sigmoid  # sigmoid归一化函数
        self.activation2 = torch.sigmoid
        self.loss = nn.functional.mse_loss  # loss函数采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.layer1(x)  # (batch_size, input_size) -> (batch_size, 1)
        x = self.activation1(x)  # (batch_size, 1) -> (batch_size, 1)   %%%%%%%%%
        x = self.layer2(x)
        y_pred = self.activation2(x)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
def build_sample():
    x = np.random.random(4)
    xlist = x.tolist()
    x.sort()
    y = [0] * 4
    y[xlist.index(x[-2])] = 1
    # if x[-1] - x[-2] < 0.2 or x[-2] - x[-3] < 0.2:
    #     return build_sample()
    # else:
    # 增加样品辨识度可训练至100%
    return xlist, y


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(Y))


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 1000
    x, y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个样本" % test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():   #上下文管理器
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比   zip交织起来
            y_p = y_p.tolist()
            y_t = y_t.tolist()
            y_p_maxi = y_p.index(max(y_p))
            y_t_maxi = y_t.index(max(y_t))
            if y_p_maxi == y_t_maxi:
                correct = correct + 1
                # print("正确！")
            else:
                wrong = wrong + 1
                # print("预测：")
                # print(y_p)
                # print("答案：")
                # print(y_t)
                # print("错误！")

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 200  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 1000  # 每轮训练总共训练的样本总数
    input_size = 4  # 输入向量维度
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        train_x, train_y = build_dataset(train_sample)  #   改成每次都用新数据训练
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            if batch_index % 2 == 0:
                optim.step()  # 更新权重
                optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:" % (epoch + 1))
        print(np.mean(watch_loss))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 画图
    # print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 4
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.47889086,0.15229675,0.31082123,0.03504317,0.18920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.78797868,0.67482528,0.13625847,0.34675372,0.09871392],
                [0.89349776,0.59416669,0.92579291,0.41567412,0.7358894]]
    # predict("model.pth", test_vec)
