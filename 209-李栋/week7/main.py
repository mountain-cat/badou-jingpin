# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import logging
import time
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data, load_vocab
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# [DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# 模型训练
def train(config):
    startTime = time.time()
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    model_loss_list = defaultdict(list)
    model_acc_list = defaultdict(list)
    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
        model_loss_list[config["model_type"]].append(np.mean(train_loss))
        model_acc_list[config["model_type"]].append(acc)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)  # 保存模型权重
    endTime = time.time()
    print("Model       Learning_Rate   Hidden_Size   Batch_Size   Pooling_Style   Epoch   Acc_Mean   Time")
    print("%s   %.5f   %d   %d   %s   %d   %.2f   %.2f" % (
        config["model_type"], config["learning_rate"], config["hidden_size"], config["batch_size"],
        config["pooling_style"], config["epoch"], np.mean(model_acc_list[config["model_type"]]),
        endTime - startTime))
    return model_loss_list, model_acc_list


# 模型预测
def predict(config):
    print("\n正在使用 %s" % config["model_type"])
    # 加载模型
    model = TorchModel(config)
    # 模型权重加载
    model.load_state_dict(torch.load('output/epoch_20.pth'))
    # state_dict = model.state_dict()

    # 打印权重
    for k, v in model.named_parameters():
        if v.requires_grad:
            print(k, v.shape)

    # 标识是否使用gpu
    # cuda_flag = torch.cuda.is_available()
    # if cuda_flag:
    #     logger.info("gpu可以使用，迁移模型至gpu")
    #     model = model.cuda()

    # 进行一次测试
    evaluator = Evaluator(config, model, logger)
    acc = evaluator.eval(20)
    return acc


if __name__ == "__main__":
    # 对比所有模型
    # 中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    model_loss_lists = defaultdict(list)
    model_acc_lists = defaultdict(list)
    for model in Config["model_type_list"]:
        Config["model_type"] = model
        for lr in [1e-5]:
            Config["learning_rate"] = lr
            for hidden_size in [128]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg"]:
                        Config["pooling_style"] = pooling_style
                        train_result = train(Config)
                        model_loss_lists[model] += train_result[0][model]
                        model_acc_lists[model] += train_result[1][model]
                        print("\n模型预测的准确率：", predict(Config))
    # 画图
    plt.rcParams['font.family'] = 'SimSun'  # 设置字体
    plt.rcParams['font.size'] = 10  # 设置字体大小

    plt.subplot(1, 2, 1)
    plt.title("model-loss均值曲线图")
    plt.xlabel("epoch")
    plt.ylabel("loss_mean")
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1
    y_major_locator = MultipleLocator(0.01)  # 把y轴的刻度间隔设置为0.01
    ax = plt.gca()  # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为0.01的倍数
    plt.xlim(0, 30)  # 把x轴的刻度范围设置为0到30
    for k, v in model_loss_lists.items():
        plt.plot(range(len(v)), [l for l in v], label=k)  # 画loss曲线
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("model-acc准确率曲线图")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1
    y_major_locator = MultipleLocator(0.01)  # 把y轴的刻度间隔设置为0.01
    ax = plt.gca()  # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为0.01的倍数
    plt.xlim(0, 30)  # 把x轴的刻度范围设置为0到30
    for k, v in model_acc_lists.items():
        plt.plot(range(len(v)), [a for a in v], label=k)  # 画acc曲线
    plt.legend()

    plt.show()
