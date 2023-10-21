# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "../data/文本分类练习.csv",
    "valid_data_path": "../data/文本分类练习.csv",
    "vocab_path": "chars.txt",
    "model_type": "gated_cnn",
    "model_type_list": ["fast_text", "cnn", "gated_cnn", "stack_gated_cnn", "rnn", "lstm", "gru", "rcnn"],
    "max_length": 20,
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 64,
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path": r"D:\NLP_CV\bert-base-chinese",
    "seed": 987,
    "class_num": 2
}
