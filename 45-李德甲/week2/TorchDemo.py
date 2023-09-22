# coding:utf8

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True' #规避OMP错误
"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个4维向量[p0,p1,p2,p3]，分成4+1种类型
1收敛型：p0小于p1,p2大于等于p0小于p1,p3大于p2小于等于p1 或者 p0大于p1,p2大于p1小于等于p0,p3小于p2大于等于p1
2延伸型：p0小于p1,p2大于p0小于p1,p3大于p1 或者 p0大于p1,p2大于p1小于p0,p3小于p1
3反向型：p0小于p1,p2小于p0,p3大于p2小于p1 或者 p0大于p1,p2大于p0,p3小于p2大于p1
4扩散型：[p0小于p1,(p2小于等于p0,p3大于p1)或者(p2小于p0,p3大于等于p1)] 
       或者 [p0大于p1,(p2大于等于p0,p3小于p1)或者(p2大于p0,p3小于等于p1)] 
0其她：非4种类型

"""


class TorchModel(nn.Module):
    def __init__(self, input_size,hidden_size,output_size):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)  # 线性层1
        self.linear2 = nn.Linear(hidden_size, hidden_size)  # 线性层2
        self.activation=torch.tanh
        self.linear3 = nn.Linear(hidden_size, hidden_size)  # 线性层2
        self.linear4 = nn.Linear(hidden_size, output_size)  # 线性层2
        self.loss = nn.CrossEntropyLoss() # loss函数采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear1(x)  # (batch_size, input_size) -> (batch_size, hidden_size)
        x = self.linear2(x)  # (batch_size, hidden_size) -> (batch_size, hidden_size)
        x=self.activation(x)
        x = self.linear3(x)
        y_pred = self.linear4(x)

        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果

def wave_type(p0,p1,p2,p3):
    if p0<p1:
        if p2>=p0 and p2<p1 and p3>p2 and p3<=p1:
            return 1
        if p2>p0 and p2<p1 and p3>p1:
            return 2
        if p2<p0 and p3>p2 and p3<p1:
            return 3
        if (p2<=p0 and p3>p1) or (p2<p0 and p3>=p1):
            return 4
        else:
            return 0
    elif p0>p1:
        if p2<=p0 and p2>p1 and p3<p2 and p3>=p1:
            return 1
        if p2<p0 and p2>p1 and p3<p1:
            return 2
        if p2>p0 and p3<p2 and p3>p1:
            return 3
        if (p2>=p0 and p3<p1) or (p2>p0 and p3<=p1):
            return 4
        else:
            return 0

# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个4维向量，通过wave_type判断类型
def build_sample():
    x = np.random.random(4)
    return x,wave_type(x[0],x[1],x[2],x[3])


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y))

#统计序列中数字的出现次数
def count_freq(nums):
    freq={}
    for num in nums:
        if num in freq:
            freq[num]+=1
        else:
            freq[num]=1
    return freq

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print(f"本次预测集类型统计：{count_freq(y.tolist())}")
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if np.argmax(y_p)==y_t:
                correct += 1  # 判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 100  # 训练轮数
    batch_size = 100  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    input_size = 4  # 输入向量维度
    hidden_size = 20 # 隐藏层维度
    output_size = 5 # 输出向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size,hidden_size,output_size)
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
            if batch_index % 2 == 0:
                optim.step()  # 更新权重
                optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 画图
    print(f'log=={log}')
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 4  # 输入向量维度
    hidden_size = 20  # 隐藏层维度
    output_size = 5  # 输出向量维度
    model = TorchModel(input_size,hidden_size,output_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%i,概率值：%s, 实际类别：%i" % (vec, np.argmax(res.tolist()), res,wave_type(vec[0], vec[1], vec[2], vec[3])))  # 打印结果

    Y = []
    for x in input_vec:
        y=wave_type(x[0], x[1], x[2], x[3])
        Y.append(y)

    correct, wrong = 0, 0
    for y_p, y_t in zip(result, Y):  # 与真实标签进行对比
            if np.argmax(y_p)==y_t:
                correct += 1  # 判断正确
            else:
                wrong += 1
    print("测试集：正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))


if __name__ == "__main__":
    main()
    test_vec = [[0.47889086,0.15229675,0.31082123,0.03504317],
                [0.94963533,0.5524256,0.95758807,0.95520434],
                [0.78797868,0.67482528,0.13625847,0.34675372],
                [0.89349776,0.59416669,0.92579291,0.41567412]]
    predict("model.pth", test_vec)
