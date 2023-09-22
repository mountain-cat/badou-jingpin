# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    #"train_data_path": "../data/train_tag_news.json",
    "train_data_path": "../data/train_customer_reviews.csv",
    #"valid_data_path": "../data/valid_tag_news.json",
    "valid_data_path": "../data/valid_customer_reviews.csv",
    "vocab_path":"chars.txt",
    #"model_type":"lstm",
    "model_type":"bert",
    "max_length": 30,
    "hidden_size": 128,
    #"hidden_size": 768,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 30,
    "batch_size": 64,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "pretrain_model_path": r"E:\works\codes\python\bert-bass-chinese\bert-base-chinese",
    "seed": 987
}