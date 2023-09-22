import torch

Config = {
    "train_file": "./data/train.json",
    "valid_file": "./data/valid.json",
    "schema_file": "./data/schema.json",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "loss_margin": 1,
    "hidden_size": 768,
    "batch_size": 256,
    "senence_max_length": 64,
    "epoch": 400,
    "learning_rate": 1e-4
}