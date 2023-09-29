import torch
import numpy as np


def softmax(x):
    exp = np.exp(x)
    return exp / exp.sum()


if __name__ == '__main__':
    x = np.array([3, 1, -3])
    print(softmax(x))

    x_tensor = torch.Tensor(x)
    print(torch.softmax(x_tensor, dim=0))
