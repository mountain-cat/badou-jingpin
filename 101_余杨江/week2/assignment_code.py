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
规律：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # 线性层
        self.activation = torch.tanh  # tanh归一化函数
        self.loss = nn.functional.mse_loss  # loss函数采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第一个值大于第五个值，认为是正样本，反之为负样本
def build_sample():
    x = np.random.choice(range(1, 6), 5)
    if x[0] > x[4]:
        return x, 1
    elif x[0] == x[4]:
        return x, 0
    else :
        return x,-1


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x) # X 存贮向量（特征）
        Y.append([y]) # Y 存贮分类（标签）
    return torch.FloatTensor(X), torch.FloatTensor(Y) #将列表 X 和 Y 转换为 PyTorch 的 FloatTensor 类型，并返回。FloatTensor 是 PyTorch 中的一种数据类型，专门用于存储浮点数，可以在 GPU 上进行加速运算。


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    # 计算三种类别的样本数量
    positive_samples = (y == 1).sum()
    zero_samples = (y == 0).sum()
    negative_samples = (y == -1).sum()
    print("本次预测集中共有%d个正样本，%d个零样本，%d个负样本" % (positive_samples, zero_samples, negative_samples ))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for i in range(len(y_pred)):
            if float(y_pred[i]) < -1/3 and int(y[i]) == -1:
                correct += 1  # 负样本判断正确
            elif float(y_pred[i]) < 1/3 and int(y[i]) == 0:
                correct += 1  # 零样本判断正确
            elif float(y_pred[i]) >= 1/3 and int(y[i]) == 1:
                correct += 1  # 正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
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
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size] #从训练集中取出当前batch的数据和对应标签。
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            if batch_index % 2 == 0: #判断当前是不是偶数batch，只有在偶数batch时才更新参数。 Why？
                optim.step()  # 更新权重
                optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测值：%f" % (vec, float(res)))  # 打印结果
        if float(res) < -1/3:
            print("预测类别：-1")
        elif float(res) < 1/3:
            print("预测类别：0")
        else:
            print("预测类别：1")


if __name__ == "__main__":
    main()
    test_vec = [[5, 4, 3, 4, 2],
                [2, 1, 5, 3, 5],
                [1, 3, 5, 4, 1],
                [2, 5, 1, 4, 2],
                [5, 1, 3, 4, 2],
                [4, 2, 5, 1, 4],
                [5, 4, 1, 3, 2],
                [4, 5, 3, 2, 4],
                [3, 2, 4, 5, 5],
                [2, 5, 3, 4, 4]]
    predict("model.pth", test_vec)
