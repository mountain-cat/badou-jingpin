# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

import random
import os
import numpy as np
import logging
from config import Config
from model import  choose_optimizer, loramodel
from evaluate import Evaluator
from loader import load_data
from peft import get_peft_model, LoraConfig, TaskType


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
    #######################################################
    ###################修改部分############################
    #######################################################
    model = loramodel
    peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"]
        )
    model = get_peft_model(model, peft_config)
    ########################################################
    
    # #加载优化器
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
            input_id, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            #######################################################
            ###################修改部分############################
            #######################################################
            output = model(input_id)[0]
            loss = nn.CrossEntropyLoss(ignore_index= -1)(output.view(-1, output.shape[-1]), labels.view(-1))
            ############################################################################
            
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
        if epoch % 20 ==1:
            microf1 , macrof1 = evaluator.get_score()
            model_path = os.path.join(config["model_path"], "epoch_%d_microf1_%f_macrof1_%f.pth" % (epoch, microf1, macrof1))
            save_tunable_parameters(model, model_path)
    return model, train_data

def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu")
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, path)
if __name__ == "__main__":
    model, train_data = main(Config)
    
"""
   最后一轮训练日志：
   2023-10-28 02:53:46,015 - __main__ - INFO - --------------------
   2023-10-28 02:53:46,016 - __main__ - INFO - epoch 100 begin
   2023-10-28 02:53:53,835 - __main__ - INFO - batch loss 0.054071
   2023-10-28 02:55:20,389 - __main__ - INFO - batch loss 0.025091
   2023-10-28 02:56:39,768 - __main__ - INFO - batch loss 0.015674
   2023-10-28 02:56:39,769 - __main__ - INFO - epoch average loss: 0.032282
   2023-10-28 02:56:39,769 - __main__ - INFO - 开始测试第100轮模型效果：
   2023-10-28 02:57:00,984 - __main__ - INFO - PERSON类实体，准确率：0.810976, 召回率: 0.678571, F1: 0.738884
   2023-10-28 02:57:00,985 - __main__ - INFO - LOCATION类实体，准确率：0.778378, 召回率: 0.587755, F1: 0.669763
   2023-10-28 02:57:00,985 - __main__ - INFO - TIME类实体，准确率：0.660131, 召回率: 0.567416, F1: 0.610267
   2023-10-28 02:57:00,985 - __main__ - INFO - ORGANIZATION类实体，准确率：0.550562, 召回率: 0.510417, F1: 0.529725
   2023-10-28 02:57:00,986 - __main__ - INFO - Macro-F1: 0.637159
   2023-10-28 02:57:00,986 - __main__ - INFO - Micro-F1 0.653900
   2023-10-28 02:57:00,986 - __main__ - INFO - --------------------
   """
