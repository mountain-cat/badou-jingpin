# coding:utf8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
         super(TorchModel, self).__init__()
         self.linear = nn.Linear(input_size, 4)  # 
         self.loss = nn.CrossEntropyLoss() 
    def forward(self, x, y=None):
        x = self.linear(x)
        if y is not None:
            return self.loss(x, y)  # 预测值和真实值计算损失
        else:
            return F.softmax(x, dim=1)  # 输出经过 softmax 处理的概率值



# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第一个值大于第五个值，认为是样本1
# 第三个值大于第二个值 认为是样本2
# 第三个值大于第四个值 认为是样本3
# ，反之为负样本
def build_sample():
    x = np.random.random(5)
    if x[0] > x[4]:
        return x, 1
    elif x[2] < x[1]:
        return x , 2
    elif x[2] > x[3]:
        return x , 3
    else:
        return x, 0


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
    print("本次预测集中共有%d个样本" % test_sample_num)
    correct = 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        predicted_labels = torch.argmax(y_pred, dim=1)  # 取概率值最大对应的类别作为预测结果
        correct = (predicted_labels == y).sum().item()  # 统计预测正确的样本个数
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / test_sample_num))
    return correct / test_sample_num




def main():
    # 配置参数
    epoch_num = 30  # 训练轮数
    batch_size = 30  # 每次训练样本个数
    train_sample = 50000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率
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
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
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
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        input_tensor = torch.FloatTensor(input_vec)
        result = model.forward(input_tensor)  # 模型预测
        _, predicted_labels = torch.max(result, 1)  # 获取预测的最大概率对应的类别索引

    for vec, res in zip(input_vec, predicted_labels):
        print("输入：%s, 预测类别：%d" % (vec, res.item()))  # 打印结果



if __name__ == "__main__":
    main()
    test_vec = [[0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                [0.66482528, 0.67482528, 0.13625847, 0.34675372, 0.67482528],
                [0.41567412, 0.59416669, 0.92579291, 0.41567412, 0.7358894],
                [0.17889086, 0.05043179, 0.19043179, 0.29229675, 0.8920843]
               ]
    predict("model.pth", test_vec)
