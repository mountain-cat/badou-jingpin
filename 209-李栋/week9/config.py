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
    "epoch": 10,
    "batch_size": 64,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": False,
    "class_num": None,
    "pretrain_model_path": r"D:\NLP_CV\bert-base-chinese"
}
