import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# 定义一个模型，用于返回向量中最大值的索引
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        # 进行5层分类
        self.linear = nn.Linear(input_size, 5)
        # 使用Softmax和交叉熵解决多分类问题
        self.activation = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss

    def forward(self, x, y=None):
        # x = self.linear(x)
        y_pred =  self.linear(x)
        # y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本，样本的生成方法，代表了我们要学习的规律
# 会从10维向量中获取数据后把数据弄成5维数据进行加法
# 最后从5为数据中选择最大的值作为真实值(Y)
def build_sample():
    x = np.random.random(10)
    res = [x[0]+x[3],x[5]+x[1],x[9]+x[2],x[7]+x[5],x[4]+x[4]]
    return x, res.index(max(res))

# 随机生成一批样本，正负样本均匀生成
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
def evaluate(model, x_test, y_test):
    model.eval()
    test_sample_num = len(y_test)
    print("本次预测集中共有%d个样本" % test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x_test)  # 模型预测
        for index, (y_p, y_t) in enumerate(zip(y_pred, y_test)):  # 与真实标签进行对比

            # 这边是为了验证模型的处理结果是不是和我估算的一样
            # print("测试集：")
            # x = x_test[index]
            # res = [x[0] + x[3], x[5] + x[1], x[9] + x[2], x[7] + x[5], x[4] + x[4]]
            # print(res)
            # print("预测值：")
            # print(y_p)
            # print("真实值：")
            # print(y_t)

            # 矩阵的最大值如果和真实值是一样代表正确
            if torch.argmax(y_p) == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / test_sample_num))
    return correct / test_sample_num


def main():
    # 配置参数
    epoch_num = 40  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    test_sample = 100 # 每轮的测试集数据
    input_size = 10  # 输入向量维度
    learning_rate = 0.001  # 学习率

    # 建立模型
    model = TorchModel(input_size)

    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 选择损失函数
    criterion = nn.CrossEntropyLoss()

    log = []

    # 创建训练集
    train_x, train_y = build_dataset(train_sample)

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]

            # 前向传播
            outputs = model(x)

            # 计算损失
            loss = criterion(outputs, y)

            # 反向传播及优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            watch_loss.append(loss.item())

        x_test, y_test = build_dataset(test_sample)
        test_accuracy = evaluate(model, x_test, y_test)
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

        # 保存每轮的loss值
        log.append([test_accuracy,float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 10
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    # print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测

        total = 0
        correct = 0
        wrong = 0
    for vec, res in zip(input_vec, result):
        input_res = [vec[0] + vec[3], vec[5] + vec[1], vec[9] + vec[2], vec[7] + vec[5], vec[4] + vec[4]]

        total += 1
        if input_res.index(max(input_res)) == torch.argmax(res).item():
            correct += 1
        else:
            wrong += 1
    print("总共数据：%d, 准确率：%f, 正确预测：%d" % (total, round(float(correct/total)), correct))  # 打印结果

if __name__ == "__main__":
    main()
    test_vec = [[0.47889086,0.15229675,0.31082123,0.03504317,0.18920843,0.78797868,0.67482528,0.13625847,0.34675372,0.09871392],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681,0.47889086,0.15229675,0.31082123,0.03504317,0.18920843],
                [0.78797868,0.67482528,0.13625847,0.34675372,0.09871392,0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.89349776,0.59416669,0.92579291,0.41567412,0.7358894,0.89349776,0.59416669,0.92579291,0.41567412,0.7358894]]
    predict("model.pth", test_vec)
