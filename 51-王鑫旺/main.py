# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
          
            input_id, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_id, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
        if epoch % 20 ==0:
            microf1 , macrof1 = evaluator.get_socre()
            model_path = os.path.join(config["model_path"], "epoch_%d_microf1_%f_macrof1_%f.pth" % (epoch, microf1, macrof1))
            torch.save(model.state_dict(), model_path)
    return model, train_data

if __name__ == "__main__":
    model, train_data = main(Config)
    
    """
    最后一轮训练日志：
    2023-09-22 12:08:04,239 - __main__ - INFO - --------------------
    2023-09-22 12:08:04,240 - __main__ - INFO - epoch 500 begin
    2023-09-22 12:08:04,714 - __main__ - INFO - batch loss 0.124191
    2023-09-22 12:08:09,973 - __main__ - INFO - batch loss 0.050561
    2023-09-22 12:08:14,909 - __main__ - INFO - batch loss 0.001766
    2023-09-22 12:08:14,910 - __main__ - INFO - epoch average loss: 0.021622
    2023-09-22 12:08:14,910 - __main__ - INFO - 开始测试第500轮模型效果：
    2023-09-22 12:08:17,216 - __main__ - INFO - PERSON类实体，准确率：0.613924, 召回率: 0.494898, F1: 0.548018
    2023-09-22 12:08:17,217 - __main__ - INFO - LOCATION类实体，准确率：0.672131, 召回率: 0.669388, F1: 0.670752
    2023-09-22 12:08:17,217 - __main__ - INFO - TIME类实体，准确率：0.810651, 召回率: 0.769663, F1: 0.789620
    2023-09-22 12:08:17,218 - __main__ - INFO - ORGANIZATION类实体，准确率：0.655172, 召回率: 0.395833, F1: 0.493502
    2023-09-22 12:08:17,218 - __main__ - INFO - Macro-F1: 0.625473
    2023-09-22 12:08:17,218 - __main__ - INFO - Micro-F1 0.648805
    2023-09-22 12:08:17,219 - __main__ - INFO - --------------------
    """
