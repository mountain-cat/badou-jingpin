import numpy as np
import torch
import torch.nn as nn


class TorchMLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(TorchMLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        out = self.linear2(x)
        return out


class MyMLP(object):

    def __init__(self, w1, w2, b1, b2):
        self.w1 = w1
        self.w2 = w2
        self.b1 = b1
        self.b2 = b2

    def forward(self, x):
        x = np.dot(x, self.w1.T) + self.b1  # 2 x 3 --> 2 x 5 torch weight default shape n_out x n_in
        out = np.dot(x, self.w2.T) + self.b2  # 2 x 5 --> 2 x 2
        return out


if __name__ == "__main__":
    np.random.seed(2023)
    features = np.random.random([2, 3]).astype("f4")  # torch默认支持float32输入
    feat_tensor = torch.from_numpy(features)
    torch_linear = TorchMLP(3, 5, 2)
    torch_pred = torch_linear.forward(feat_tensor)
    print(torch_pred.data)
    print("=" * 50)

    state_dict = torch_linear.state_dict()
    w1 = state_dict.get("linear1.weight").numpy()
    w2 = state_dict.get("linear2.weight").numpy()
    b1 = state_dict.get("linear1.bias").numpy()
    b2 = state_dict.get("linear2.bias").numpy()

    my_linear = MyMLP(w1, w2, b1, b2)
    my_pred = my_linear.forward(features)
    print(my_pred)
