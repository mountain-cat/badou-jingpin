# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path":"F:\\AiModels\\bert-base-chinese\\vocab.txt",
    "max_length": 150,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 64,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": True,
    "class_num": 9,
    "pre_train_model_path": "F:\\AiModels\\bert-base-chinese"
}