# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，有多少个>0.5的x，类别就是多少

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        # super(TorchModel, self).__init__()
        super().__init__()
        # 线性层， 输出维度和输入维度
        self.linear = nn.Linear(input_size, 1)
        # # sigmoid归一化函数
        # self.activation = torch.sigmoid
        # loss函数采用均方差损失mse_loss
        self.loss = nn.functional.mse_loss

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # (batch_size, input_size) -> (batch_size, 1)
        x = self.linear(x)
        # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            # 预测值和真实值计算损失
            return self.loss(x, y)
        else:
            # 输出预测结果
            return x


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
def build_sample():
    x = np.random.random(5)
    return x, sum(x > 0.5)


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.FloatTensor(Y), Y


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 100  # 每次训练样本个数
    train_sample = 100000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 创建训练集，正常任务是读取训练集
    train_x, train_y, Y = build_dataset(train_sample)
    print(f"本次训练集中的样本标签分布：\n{pd.Series(Y).value_counts()}")
    # 训练过程
    for epoch in range(epoch_num):
        # 训练模式
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            # 计算loss
            loss = model(x, y)
            # 计算梯度
            loss.backward()
            # 更新权重
            optim.step()
            # 梯度归零
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("===========================\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    # 保存模型
    torch.save(model.state_dict(), "model.pt")
    return None


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print()
    print(model.state_dict())
    print()
    # 测试模式
    model.eval()
    # 在测试过程中关闭梯度计算和 Batch Normalization 的运算，并保证测试数据和训练数据的统计特征相同
    with torch.no_grad():
        # 模型预测结果
        result = model.forward(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec, result):
        print(f"输入：{vec}，真实类别：{sum(np.array(vec) > 0.5)}， 预测类别：{round(float(res))}， 预测值：{res}")


if __name__ == "__main__":
    main()
    test_vec = []
    for num in range(5):
        test_vec.append(list(np.random.random(5)))
    predict("model.pt", test_vec)
