# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "data/schema.json",
    "train_data_path": "data/train_corpus",
    "valid_data_path": "data/valid_corpus",
    "vocab_path": "F:\\AiModels\\bert-base-chinese\\vocab.txt",
    "max_length": 50,
    "hidden_size": 128,
    "epoch": 10,
    "batch_size": 128,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": True,
    "class_num": 4,
    "vocab_size": 3000,
    "pre_train_model_path": "F:\\AiModels\\bert-base-chinese"
}