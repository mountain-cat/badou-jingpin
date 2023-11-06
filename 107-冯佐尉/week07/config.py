import torch

Config = {
    "file_path": "text.csv",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "input_dim": 768,
    "hidden_dim": 1024,
    "lstm_layers": 2,
    "output_dim": 2,
    "batch_size": 300,
    "senence_max_length": 512,
    "epoch": 200,
    "learning_rate": 1e-3
}