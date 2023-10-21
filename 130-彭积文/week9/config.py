import torch

Config = {
    "train_file": "./ner_data/train",
    "valid_file": "./ner_data/test",
    "schema_file": "./ner_data/schema.json",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "hidden_size": 768,
    "type_num": 9,
    "batch_size": 256,
    "sentence_max_length": 128,
    "epoch": 100,
    "learning_rate": 1e-4
}