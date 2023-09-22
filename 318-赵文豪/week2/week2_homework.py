import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plot

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个6维向量，如果同时满足第1个数>第2个数、第3个数>第4个数、第5个数>第6个数三个条件，则归为第1类
                  如果只满足其中两个条件则归为第2类
                  如果只满足其中1个条件则归为第3类
                  若都不满足则归为第4类
"""


class TorchModel(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, out_size):
        super(TorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden1_size)  # 第一层线性层
        self.activation1 = nn.GELU()  # 第一层激活层
        self.layer2 = nn.Linear(hidden1_size, hidden2_size)  # 第二层线性层
        self.activation2 = nn.GELU()  # 第二层激活层
        self.layer3 = nn.Linear(hidden2_size, out_size)  # 第三层线性层
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x1 = self.layer1(x)
        x1 = self.activation1(x1)
        x2 = self.layer2(x1)
        x2 = self.activation2(x2)
        y_pre = self.layer3(x2)
        if y is not None:
            return self.loss(y_pre, y)
        else:
            return y_pre
            # return nn.Softmax(y_pre)


# 生成一组样本
def build_sample():
    x = np.random.random(6)
    num = 0
    if x[0] > x[1]:
        num += 1
    if x[2] > x[3]:
        num += 1
    if x[4] > x[5]:
        num += 1
    y = np.zeros(4, dtype=int)
    y[3 - num] = 1
    return x, y


# 生成一批样本
def build_dataset(total_sample_num):
    data_x = []
    data_y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        data_x.append(x)
        data_y.append(y)
    return torch.FloatTensor(data_x), torch.FloatTensor(data_y)
    # return X, Y


# 准确率计算
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pre = model(x)
        for y_t, y_p in zip(y, y_pre):
            if y_t.argmax() != y_p.argmax():
                wrong += 1
            else:
                correct += 1
    return correct / (correct + wrong)


# 数据训练
def main():
    # 配置参数
    train_sample_num = 5000
    epoch_num = 20
    batch_size = 50
    learning_rate = 0.01
    # 建立模型
    model = TorchModel(6, 12, 8, 4)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 采用优化器中的Adam模型
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample_num)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()  # 进入训练模式
        watch_loss = []
        for batch in range(train_sample_num // batch_size):
            x = train_x[epoch * batch_size:(epoch + 1) * batch_size]
            y = train_y[epoch * batch_size:(epoch + 1) * batch_size]
            loss = model(x, y)
            optim.zero_grad()  # 梯度归零
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, float(np.mean(watch_loss))))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    plot.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plot.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plot.legend()
    plot.show()
    return


# 数据预测
def predict(model_path, input_vec):
    model = TorchModel(6, 12, 8, 4)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d" % (vec, res.argmax() + 1))  # 打印结果


# 主函数
if __name__ == "__main__":
    main()
    test_vec = [[0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843, 0.51082123],
                [0.74963533, 0.5524256, 0.95758807, 0.61520434, 0.74890681, 0.95758807],
                [0.78797868, 0.47482528, 0.13625847, 0.34675372, 0.09871392, 0.67482528],
                [0.70347868, 0.59417868, 0.92242591, 0.45562425, 0.7858894, 0.50344256],
                [0.30344788, 0.59416669, 0.22579291, 0.41858894, 0.6358894, 0.80349776],
                [0.80342528, 0.59225769, 0.92579291, 0.41967512, 0.8358894, 0.70340434],
                [0.40349776, 0.59412528, 0.22675391, 0.41567412, 0.5358894, 0.70825286],
                ]
    predict("model.pth", test_vec)
