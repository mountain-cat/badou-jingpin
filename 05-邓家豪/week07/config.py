# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "文本分类练习数据集/文本分类练习train.csv",
    "valid_data_path": "文本分类练习数据集/文本分类练习predict.csv",
    "vocab_path": "chars.txt",
    "model_type": "lstm",
    "max_length": 20,
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 64,
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path": r"/Users/jeromedenn/学习/badou-jingpin/05-邓家豪/week07/bert-base-chinese",
    "seed": 987,
    "class_num": 2
}
