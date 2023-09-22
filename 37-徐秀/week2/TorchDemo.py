# coding:utf8

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import torch.nn.functional as Fun


"""

基于pytorch框架编写鸢尾花多分类模型并训练

"""
iris = datasets.load_iris()
input_data = torch.FloatTensor(iris['data'])
label = torch.LongTensor(iris['target'])


# 构建模型
class TorchModel(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(TorchModel, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.out = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        """前向传播"""
        x = Fun.relu(self.hidden(x))
        x = self.out(x)
        return x


def main():
    # 配置参数
    epoch_num = 200  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = input_data.shape[0]  # 每轮训练总共训练的样本总数
    input_size = 4  # 输入的特征维度
    output_size = 3  # 输出的类别个数
    n_hidden = 20  # 神经元个数
    learning_rate = 0.001  # 学习率

    # 建立模型
    model = TorchModel(input_size, n_hidden, output_size)
    # 选择优化器，model.parameters()优化的参数
    optim = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()  # 交叉熵损失
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            y = input_data[batch_index * batch_size: (batch_index + 1) * batch_size]
            label1 = label[batch_index * batch_size: (batch_index + 1) * batch_size]
            out = model(y)
            # 计算损失
            loss = loss_func(out, label1)
            loss.backward()  # 计算梯度
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        out = model(input_data)
        pre = torch.max(out, 1)[1]
        pre_y = pre.data.numpy()
        targe_y = label.data.numpy()
        acc = float((pre_y == targe_y).astype(int).sum()) / float(targe_y.size)
        print("鸢尾花预测准确率{}".format(acc))
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
    output_size = 3
    n_hidden = 20
    model = TorchModel(input_size, n_hidden, output_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = torch.max(model.forward(torch.FloatTensor(input_vec)), 1)  # 模型预测
    i = 1
    for vec, type, res in zip(input_vec, result[1], result[0]):
        print("num %i,输入：%s, 预测类别：%d, 概率值：%f" % (i, vec, type, res))  # 打印结果
        i += 1


if __name__ == "__main__":
    main()
    test_vec = [[5.1, 3.5, 1.4, 0.2],
                [4.9, 3., 1.4, 0.2],
                [7.7, 3.8, 6.7, 2.2],
                [6.4, 2.8, 5.6, 2.1],
                [7.2, 3., 5.8, 1.6],
                [5.9, 3., 5.1, 1.8]]
    predict("model.pth", test_vec)
