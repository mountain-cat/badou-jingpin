# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "nn_pipeline/output",
    "summary_table_path":"summary_table.csv",
    "raw_data_path": "data/文本分类练习.csv",
    "train_data_path": "data/train_tag_news.json",
    "valid_data_path": "data/valid_tag_news.json",
    "vocab_path": "nn_pipeline/chars.txt",
    "model_type": "bert",
    "max_length": 25,
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 64,
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path": r"C:\Users\zhang\Desktop\week作业\badou-jingpin\24-张材渊\week6\bert-base-chinese",
    "seed": 987
}
