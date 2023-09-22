import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 输入一个10纬度向量，第1个数大于从左第二开始算起第几个数就是第几分类，如果都不大于属于0分类：以0下标开始计数
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        size = 128
        self.linear1 = nn.Linear(input_size, size)
        self.ac = torch.relu
        self.linear = nn.Linear(size, 10)
        self.loss = nn.CrossEntropyLoss()
    def forward(self, x, y=None):
        x = self.linear1(x)
        x = self.ac(x)
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

def build_sample():
    x = np.random.random(10)
    y = 0
    for index, val in enumerate(x):
        if index > 0 and x[0] > val:
            y = index
    return x, y

def build_datas(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


def main():
    epoch_num = 10
    batch_size = 20
    train_sample = 50000
    input_size = 10      # 输入维度
    learning_rate = 0.01
    # 建立模型
    model = TorchModel(input_size)
    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建数据集
    train_x, train_y = build_datas(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for index in range(train_sample // batch_size):
            x = train_x[index*batch_size:(index+1)*batch_size]
            y = train_y[index * batch_size:(index + 1) * batch_size:]
            # 计算loss
            loss = model(x, y)
            # 计算梯度
            loss.backward()
            # 更新权重
            optim.step()
            # 梯度归零
            optim.zero_grad()
            watch_loss.append(loss.item())
        # 正确校验
        acc = evaluate(model)
        # np.mean求平均值
        print("第%d轮平均loss： %f"%(epoch+1, np.mean(watch_loss)))
        log.append([acc, float(np.mean(watch_loss))])
    torch.save(model.state_dict(),"modelt.pth")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

# 计算准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_datas(test_sample_num)
    # print("本次样本总数%d: 正样本有%d,副样本有%d"%(test_sample_num, sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    # 忽略梯度
    with torch.no_grad():
        pred_y = model(x)
        for y_p, y_t in zip(pred_y, y):
            y_p = y_p.tolist()
            y_t = y_t.tolist()
            if y_p.index(max(y_p)) == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct + wrong)))
    return correct/(correct + wrong)





# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 10
    # 调用模型
    model = TorchModel(input_size)
    # 模型添加权重
    model.load_state_dict(torch.load(model_path))
    print("模型权重:", model.state_dict())
    # 测试模型
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
        for vec, res in zip(input_vec, result):
            # 打印结果
            print(result)
            print("输入：%s, 预测类别：%d， 概率值: %f"%(vec, round(float(res)), res))

if __name__ == "__main__":
    main()
    test_vec = [[0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843,0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                [0.94963533, 0.9524256, 0.95758807, 0.95520434, 0.99890681,0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                [0.78797868, 0.67482528, 0.13625847, 0.34675372, 0.09871392,0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                [0.89349776, 0.99416669, 0.92579291, 0.41567412, 0.7358894,0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],[0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843,0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                [0.94963533, 0.9524256, 0.95758807, 0.95520434, 0.99890681,0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                [0.78797868, 0.67482528, 0.13625847, 0.34675372, 0.09871392,0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                [0.89349776, 0.99416669, 0.92579291, 0.41567412, 0.7358894,0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843]]
    predict("modelt.pth", test_vec)

