# coding:utf8

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import model, dataloader
from sklearn.metrics import classification_report
device = torch.device('cuda:0')


def main():
    model_1 = None
    # 配置参数
    train_percent = 0.8 # 训练集占数据集的百分比
    batch_size = 128
    epoch_num = 200  # 训练轮数
    learning_rate = 0.000001  # 学习率
    mode = 0 # 0:训练网络, 1:读取本地模型训练或预测, 2:计算模型大小和FLOPs
    mode_ = model.Bert()
    target_names = ['class 0', 'class 1']
    label = [0, 1]
    # 建立模型
    if mode == 0:
        model_1 = mode_.to(device)
    else:
        model_1.load_state_dict(torch.load('model.pth'))
        model_1 = model_1.to(device)

    # 选择优化器
    optim = torch.optim.AdamW(model_1.parameters(), lr=learning_rate)
    log = []
    dataset, num = dataloader.get_data(batch_size)
    # 训练过程
    for epoch in range(epoch_num):
        model_1.train()
        eval_num = 0
        acc_num = 0
        watch_loss = []
        acc_max = 0
        for i, (x, y) in enumerate(tqdm(dataset)):
            y1 = y
            x = x.to(device)
            y = y.to(device)
            if i <= num * train_percent:
                outputs, loss = model_1(x, y)
                loss.backward()
                watch_loss.append(loss.item())
                optim.step()
                optim.zero_grad()
                outputs = outputs.to('cpu').detach().numpy()
                y = y.to('cpu').detach().numpy()
                out = []
                for k in (outputs.reshape(-1)):
                    # 移动阈值
                    if k >= 0.53:
                        out.append(1)
                    else:
                        out.append(0)
                out = np.array(out).reshape(-1)
                acc_num += np.sum(out == y, dtype=np.float)
                if i % 50 == 0 and not i == 0:
                    print("第%d轮训练集平均loss:%f, 平均准确率:%f" % (epoch + 1, np.mean(watch_loss), 100 * (acc_num/(i*batch_size))))
                watch_loss.append(loss.item())
                i1 = i

            elif i > num * train_percent:
                if eval_num == 0:
                    acc_num = 0
                    model_1.eval()
                    eval_num += 1
                    watch_loss = []
                    y_true = []
                    y_pred = []
                outputs = model_1(x)
                y_true.append(y1.to('cpu').detach().numpy())
                outputs = outputs.to('cpu').detach().numpy()
                out = []
                for k in (outputs.reshape(-1)):
                    # 移动阈值
                    if k >= 0.53:
                        out.append(1)
                    else:
                        out.append(0)
                out = np.array(out).reshape(-1)
                y_pred.append(out)
        print("############第%d轮测试集平均准确率:############" % (epoch + 1))
        y_true1 = np.array(y_true).reshape(-1)
        y_pred1 = np.array(y_pred).reshape(-1)
        print(classification_report(y_true1, y_pred1, labels=label, target_names=target_names, digits=4))
        if 100 * (acc_num/((i-i1) * batch_size)) < 100.0:
            acc = 100 * (acc_num / ((i - i1) * batch_size))
        if acc_max < 100 * (acc_num / (num * (1-train_percent) * batch_size)):
            acc_max = 100 * (acc_num / (num * (1-train_percent) * batch_size))
            torch.save(model_1.state_dict(), "model_mine_PCB-TP.pth")
    learning_rate *= 0.99
    optim = torch.optim.Adam(model_1.parameters(), lr=learning_rate)
    return


softmax = nn.Softmax(dim=-1)


# 使用训练好的模型做预测
def predict(model_path, test_strings, vocab_path="vocab.json"):
    input_size = 64
    total_num = 20
    hidden_size = 128
    acc_num = 0
    data = []
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    for i in test_strings:
        data.append([vocab.get(word, vocab['unk']) for word in i])
    data_feed = torch.LongTensor(np.array(data))
    model_1 = model.Model(input_size, vocab, hidden_size).to(device) # 加载模型
    model_1.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    model_1.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        for i, x in enumerate(data_feed):
            result = model_1.forward(x.to(device))  # 模型预测
            print("输入参数:", test_strings[i])
            print("预测结果:", softmax(result)[0])
            print("预测类别:%d, 置信度:%f." % (torch.argmax(result.data, -1)[0], softmax(result)[0][torch.argmax(result.data, -1).item()]))
            print("======================================")


if __name__ == "__main__":
    train = 1 # train=1训练网络， train=0预测数据
    if train:
        main()
    else:
        test_strings = [""]
        predict("model.pth", test_strings)
