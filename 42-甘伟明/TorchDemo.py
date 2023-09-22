# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import data_generator
import matplotlib.pyplot as plt
device = torch.device('cuda:0')
"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，
    if x[0] > x[4] and x[1] > x[2]:
        return x, 0
    elif x[0] > x[4] and x[1] < x[2]:
        return x, 1
    elif x[0] < x[4] and x[1] > x[2]:
        return x, 2
    elif x[0] < x[4] and x[1] < x[2]:
        return x, 3
"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 16)
        self.linear1 = nn.Linear(16, 4)
        self.activation = torch.nn.ReLU()
        self.Dropout = torch.nn.Dropout(p=0.00)
        self.bn = torch.nn.LazyBatchNorm1d()
        self.loss = nn.CrossEntropyLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)
        x = self.Dropout(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.linear1(x)
        if y is not None:
            return x, self.loss(x, y[:, 0].long())
        else:
            return x


def main():
    # 配置参数
    total_num = 6000 # 总体样本数
    train_percent = 0.8 # 训练集占数据集的百分比
    batch_size = 16
    epoch_num = 100  # 训练轮数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    mode = 0
    # 建立模型
    if mode == 0:
        model = TorchModel(input_size).to(device)
    else:
        model = TorchModel(input_size)
        model.Dropout = torch.nn.Dropout(p=0.00)
        model.load_state_dict(torch.load('model.pth'))
        model = model.to(device)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    dataset, num = data_generator.get_data(total_num=total_num, batch_size=batch_size)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        eval_num = 0
        acc_num = 0
        watch_loss = []
        acc_max = 0
        for i, (x, y) in enumerate(dataset):
            x = x.to(device)
            y = y.to(device)
            if i <= num * train_percent:
                outputs, loss = model(x, y)  # 计算loss
                loss.backward()  # 计算梯度
                watch_loss.append(loss.item())
                optim.step()  # 更新权重
                optim.zero_grad()  # 梯度归零
                acc_num += torch.sum(torch.argmax(outputs.data, -1) == y[:, 0], dtype=torch.float32).item()
                if i % 40 == 0 and not i==0:
                    print("第%d轮训练集平均loss:%f, 平均准确率:%f" % (epoch + 1, np.mean(watch_loss), 100 * (acc_num/((i+1)*batch_size))))
            elif i > num * train_percent:
                if eval_num == 0:
                    acc_num = 0
                    model.eval()
                    eval_num += 1
                    watch_loss = []
                outputs = model(x)  # 计算loss
                # print(torch.argmax(outputs.data, -1).shape, y[:, 0].shape)
                acc_num += torch.sum(torch.argmax(outputs.data, -1) == y[:, 0], dtype=torch.float32).item()
                watch_loss.append(loss.item())
        # print(torch.argmax(outputs.data, -1), y)
        print("############第%d轮测试集平均loss:%f, 平均准确率:%f############" % (epoch + 1, np.mean(watch_loss), 100 * (acc_num/(num*(1-train_percent) * batch_size))))
        log.append([100 * (acc_num/(num * (1-train_percent) * batch_size)), float(np.mean(watch_loss))])
        if acc_max < 100 * (acc_num / (num * (1-train_percent) * batch_size)):
            acc_max = 100 * (acc_num / (num * (1-train_percent) * batch_size))
            torch.save(model.state_dict(), "model.pth")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


softmax = nn.Softmax(dim=-1)


# 使用训练好的模型做预测
def predict(model_path):
    input_size = 5
    total_num = 20
    acc_num = 0
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    dataset, num = data_generator.get_data(total_num=total_num, batch_size=1)
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        for i, (x, y) in enumerate(dataset):
            result = model.forward(x)  # 模型预测
            print("输入参数:", x[0])
            print("预测结果:", softmax(result)[0])
            print("预测类别:%d, 标签类别:%d,置信度:%f." % (torch.argmax(result.data, -1)[0], y[0], softmax(result)[0][torch.argmax(result.data, -1).item()]))
            print("======================================")
            acc_num += torch.sum(torch.argmax(result.data, -1) == y[:, 0], dtype=torch.float32).item()
        print("总体准确率:%d" % (acc_num / total_num * 100))  # 打印结果


if __name__ == "__main__":
    """
    生成数据的规则如下：
    if x[0] > x[4] and x[1] > x[2]:
        return x, 0
    elif x[0] > x[4] and x[1] < x[2]:
        return x, 1
    elif x[0] < x[4] and x[1] > x[2]:
        return x, 2
    elif x[0] < x[4] and x[1] < x[2]:
        return x, 3
    """
    train = 0 # train=1训练网络， train=0预测数据
    if train:
        main()
    else:
        predict("model.pth")
