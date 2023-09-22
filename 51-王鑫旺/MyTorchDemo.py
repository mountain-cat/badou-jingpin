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


"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 2*input_size)  # 线性层
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(2*input_size, 5)  # 线性层
        self.loss = nn.CrossEntropyLoss()  # loss函数采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear1(x)  # (batch_size, input_size) -> (batch_size, 1)
        x = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        x = self.linear2(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第一个值大于第五个值，认为是正样本，反之为负样本
def decide(x):
    a = x[0] > x[4]
    b = x[1] > x[4]
    c = x[2] > x[4]
    d = x[3] > x[4]
    e = [a, b, c, d]
    count_true = e.count(True)
    if count_true == 1:
        return x, count_true
    elif count_true == 2:
        return x, count_true
    elif count_true == 3:
        return x, count_true
    elif count_true == 4:
        return x, count_true
    else:
        return x, 0

def build_sample():
    x = np.random.random(5)
    a,b=decide(x)
    return a,b

def to_one_hot(target, shape=5):
    one_hot_target = np.zeros(shape)

    one_hot_target[target] = 1
    return one_hot_target
# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(to_one_hot(y))
    return torch.FloatTensor(X), torch.FloatTensor(Y)

def one_hot(target):
    max_index = target.argmax()
    result = torch.zeros_like(target)
    result[max_index] = 1
    result=torch.tensor(result,dtype=torch.float)
    return result


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            #print(y_p,y_t)

            result = one_hot(y_p)

            if torch.equal(result,y_t):
                correct += 1  # 负样本判断正确
            else:
                    wrong += 1

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 500  # 每次训练样本个数
    train_sample = 100000  # 每轮训练总共训练的样本总数
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
    # 画图
    # print(log)
    # plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    # plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    # plt.legend()
    # plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    #print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):

        v,l=decide(vec)
        tst=to_one_hot(l)
        tst=torch.tensor(tst,dtype=torch.float)
        pre=one_hot(res)
        end=torch.equal(tst,pre)
        print("输入：%s, 转为onehot：%s, 预测：%s, 预测正确：%r" % (vec, tst, pre, end))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.47889086,0.15229675,0.31082123,0.03504317,0.18920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.78797868,0.67482528,0.13625847,0.34675372,0.09871392],
                [0.89349776,0.59416669,0.92579291,0.41567412,0.7358894]]
    predict("model.pth", test_vec)
