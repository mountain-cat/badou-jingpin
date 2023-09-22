# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "../data/schema.json",
    "train_data_path": "../data/train.json",
    "valid_data_path": "../data/valid.json",
    "vocab_path": "../chars.txt",
    "max_length": 20,  # 文本的最大长度
    "hidden_size": 1024,
    "epoch": 40,
    "batch_size": 128,
    "epoch_data_size": 1280,  # 每轮训练中采样数量
    "positive_sample_rate": 0.5,  # 正样本比例，通常希望正负样本比例是均衡的
    "optimizer": "adam",
    "learning_rate": 1e-3,
}
