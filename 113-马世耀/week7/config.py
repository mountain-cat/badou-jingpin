#-*-coding:utf-8-*-
'''
Author: Shiyao Ma
Date: 2023-09-01 15:43:50
LastEditors: Shiyao Ma
LastEditTime: 2023-09-02 11:54:39
Copyright (c) 2023 by Shiyao Ma, All Rights Reserved. 
'''
# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "train.json",
    "valid_data_path": "valid.json",
    "vocab_path":"chars.txt",
    "model_type":"lstm",
    "max_length": 70,
    "hidden_size": 128,
    # "kernel_size": 3,
    "num_layers": 2,
    "epoch": 50,
    "batch_size": 64,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"/home/mashiyao/.cache/huggingface/hub/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55/",
    "seed": 987,
    "train_size": 0.8
}