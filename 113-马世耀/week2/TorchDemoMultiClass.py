#-*-coding:utf-8-*-
'''
Author: Shiyao Ma
Date: 2023-07-28 11:17:30
LastEditors: Shiyao Ma
LastEditTime: 2023-07-31 11:04:24
Copyright (c) 2023 by Shiyao Ma, All Rights Reserved. 
'''
import os
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

# Initialization ----
## path
os.chdir(osp.dirname(osp.abspath(__file__)))
print(f"Working under {os.getcwd()}")

## cuda device
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda")
    print("Running on GPU, hard set to device '0'")
else:
    device = torch.device("cpu")
    print("Running on CPU")

## rng seed
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


"""
@mashiyao
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，一共3个类别，如果第1个数为最大则为类别1，如果第1个数为最小则为类别2，其余情况则为类别3
理论上，生成的标签中类别0:1:2 ~= 1:1:3，可设置balanced = True对生成样本的类别平衡
"""


class TorchModelMultiClass(nn.Module):
    def __init__(self, input_size, hidden_size: int = 50, num_classes: int = 3):
        super(TorchModelMultiClass, self).__init__()
        self.num_classes = num_classes
        self.linear = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
            nn.Linear(hidden_size, self.num_classes),
            nn.Softmax()
        )
        self.criterion = nn.CrossEntropyLoss() # loss函数采用交叉熵
        self._init_weights([self.linear])

    def _init_weights(self, blocks):
        """ Normal initialization """
        for block in blocks:
            for m in block:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
        
    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x) # (batch_size, input_size) -> (batch_size, num_classes)
        if y is not None:
            y_onehot = nn.functional.one_hot(y, num_classes=self.num_classes)
            return self.criterion(y_pred, y_onehot.to(torch.float32)) # 预测值和真实值计算损失
        else:
            return y_pred # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
def build_sample():
    """ Generate vector with 5 randomly distributed number from Unif(0~1] dist.
    If the first element is the highest, assign label [0] to the vector
    If the first element is the lowest, assign label [1] to the vector
    If else, assign label [2] to the vector
    """
    x = np.random.random(5)
    # if np.max(x) == x[0]:
    #     return x, 0
    # elif np.max(x) == x[1]:
    #     return x, 1
    # else:
    #     return x, 2
    if np.max(x) == x[0]:
        return x, 0
    elif np.min(x) == x[0]:
        return x, 1
    else:
        return x, 2


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num, balanced: bool = True):
    """ Sample constructing, and analyzing label distributions. """
    print(f"Start generating dataset of {total_sample_num} samples...")
    X = []
    Y = []
    total_num = 0
    counter = {'0': 0, '1': 0, '2': 0}
    threshold = total_sample_num // 3 if balanced else total_sample_num
    while total_num < total_sample_num:
        x, y = build_sample()
        if counter[str(y)] > threshold:
            continue
        X.append(x)
        Y.append(y)
        counter[str(y)] += 1
        total_num += 1
    print(f"""[{'balanced' if balanced else 'inbalanced'}] Sample generated. Label distributed as follows:
    0: {counter['0']}
    1: {counter['1']}
    2: {counter['2']}
    """)
    return torch.FloatTensor(X), torch.tensor(Y, dtype=torch.int64)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num, balanced=False)
    x = x.to(device)
    y = y.to(device)
    # print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t, x_i in zip(y_pred, y, x):  # 与真实标签进行对比
            y_p_offload, y_t_offload = y_p.cpu().numpy(), y_t.cpu().numpy()
            if np.argmax(y_p_offload).item() == int(y_t_offload):
                correct += 1  # 负样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 40  # 训练轮数
    batch_size = 25  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01 # 学习率
    # 建立模型
    model = TorchModelMultiClass(input_size)
    model.to(device)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
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
    torch.save(model.state_dict(), "model\\model.pth")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModelMultiClass(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        gt = -1
        if vec[0] == np.max(vec):
            gt = 0
        elif vec[0] == np.min(vec):
            gt = 1
        else:
            gt = 2
        print("输入：%s, 预测类别：%d, 真实类别：%d, 概率值：%f" % (vec, np.argmax(res.cpu().numpy()).item(), gt, res.cpu().numpy().max().item()))  # 打印结果


if __name__ == "__main__":
    set_seed()
    main()
    test_vec = [[0.47889086,0.15229675,0.31082123,0.03504317,0.18920843],   # gt: 0
                [0.24963533,0.5524256,0.95758807,0.95520434,0.84890681],    # gt: 1
                [0.78797868,0.67482528,0.13625847,0.34675372,0.09871392],   # gt: 0
                [0.69349776,0.99416669,0.92579291,0.41567412,0.7358894],    # gt: 2
                [0,1,2,3,4],                                                # gt: 1
                [2,1,0,3,4],                                                # gt: 2
                [4,0,1,2,3]]                                                # gt: 0
    predict("model\\model.pth", test_vec)
