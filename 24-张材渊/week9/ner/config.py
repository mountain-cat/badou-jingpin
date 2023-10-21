# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train.txt",
    "valid_data_path": "ner_data/test.txt",
    "vocab_path": "chars.txt",
    "max_length": 150,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 64,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": False,
    "class_num": 9,
    "model_type": "lstm",
    "pretrain_model_path": r"C:\Users\zhang\Desktop\week作业\badou-jingpin\24-张材渊\week6\bert-base-chinese",
}
