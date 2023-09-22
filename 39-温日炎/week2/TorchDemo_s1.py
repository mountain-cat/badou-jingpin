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
规律：x是一个5维向量，
如果第1个数>第5个数，则为1类样本，
如果第1个数>第4个数，则为2类样本，
如果第1个数>第3个数，则为3类样本，
如果第1个数>第2个数，则为4类样本
其它为5类样本

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 4)  # 线性层
        self.activation = torch.nn.Softmax  # sigmoid归一化函数，激活函数
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        # print(x)
        y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        # print(y_pred.dim)
        if y is not None:
            return self.loss(y_pred.dim, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 如果第1个数>第5个数，则为1类样本，
# 如果第1个数>第4个数，则为2类样本，
# 如果第1个数>第3个数，则为3类样本，
# 如果第1个数>第2个数，则为4类样本
# 其它为5类样本
def build_sample():
    x = np.random.random(5)
    if x[0] > x[4]:
        return x, 0
    elif x[0] > x[3]:
        return x, 1
    elif x[0] > x[2]:
        return x, 2
    else:
        return x, 3


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("一类样本：%d,二类样本：%d,三类样本：%d，四类样本：%d" % (sum(y.eq(0)), sum(y.eq(1)), sum(y.eq(2)), sum(y.eq(3))))
    # print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    # type1, type2, type3, type4, type5, worng = 0, 0, 0, 0, 0, 0
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        # print(y_pred.dim)
        y_pred = y_pred.dim
        for y_p, y_t in zip(torch.tensor(y_pred), y):
            if y_t == y_p.argmax():
                correct += 1;
            else:
                wrong += 1
        return correct/(correct+wrong)


def main():
    # 配置参数
    epoch_num = 10000  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 1000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.1  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # print(model)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # print(train_x, train_y)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        # 创建训练集，正常任务是读取训练集
        train_x, train_y = build_dataset(train_sample)
        # print(train_x, train_y)
        loss = model(train_x, train_y)  # 计算loss
        loss.backward()  # 计算梯度
        optim.step()  # 更新权重
        optim.zero_grad()  # 梯度归零
        watch_loss.append(loss)
        acc = evaluate(model)  # 测试本轮模型结果
        print("acc%d", acc)
        # print(acc)
        # log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    # print("loss%d", watch_loss)
    torch.save(model.state_dict(), "model2.pth")
    # 画图
    # print(log)
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
    model.state_dict()

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    print(result)


if __name__ == "__main__":
    main()
    test_vec = [[0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.78797868, 0.67482528, 0.13625847, 0.34675372, 0.09871392],
                [0.89349776, 0.59416669, 0.92579291, 0.41567412, 0.7358894]]
    predict("model2.pth", test_vec)
