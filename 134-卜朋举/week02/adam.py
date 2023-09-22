import copy

import numpy as np
import torch
import torch.nn as nn


class TorchLinearReg(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(TorchLinearReg, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss()

    def forward(self, x, y=None):
        x = self.linear(x)
        x = self.sigmoid(x)
        if y is not None:
            return self.loss(x, y)
        else:
            return x


class MyLinearReg(object):

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, x, y=None):
        x = np.dot(self.weight, x)
        x = self.sigmoid(x)
        if y is not None:
            return self.mse(x, y)
        else:
            return x

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def mse(self, y_pred, y_true):
        return np.sum((y_pred - y_true) ** 2) / len(y_pred)

    def backward(self, y_pred, y_true, x):
        grad_loss = 2 / len(y_pred) * (y_pred - y_true)
        grad_sigmoid = y_pred * (1 - y_pred)
        grad_w = x
        grad = grad_loss * grad_sigmoid * grad_w
        return grad


def my_sgd(lr, w, w_grad):
    return w - lr * w_grad


def my_adam(lr, w, w_grad):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-3
    t = 0
    mt = 0
    vt = 0

    t += 1
    gt = w_grad
    mt = beta1 * mt + (1 - beta2) * gt
    vt = beta2 * vt + (1 - beta1) * gt ** 2
    mth = mt / (1 - beta1)
    vth = vt / (1 - beta2)
    w = w - lr * mth / (np.sqrt(vth) + eps)
    return w


if __name__ == '__main__':
    x = np.array([1, 2, 3, 4])
    y = np.array([0.1, -0.1, 0.01, -0.01])

    torch_model = TorchLinearReg(len(x), len(x))
    torch_w = torch_model.state_dict()["linear.weight"]
    npy_w = copy.deepcopy(torch_w.numpy())
    x_tensor = torch.from_numpy(x).float().unsqueeze(0)
    y_tensor = torch.from_numpy(y).float().unsqueeze(0)
    torch_loss = torch_model(x_tensor, y_tensor)
    print("torch loss: ", torch_loss)

    my_model = MyLinearReg(npy_w)
    my_loss = my_model(x, y)
    print("my loss: ", my_loss)

    lr = 0.01
    # optimizer = torch.optim.SGD(torch_model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=lr)
    optimizer.zero_grad()

    torch_loss.backward()
    print("torch grad: ", torch_model.linear.weight.grad)

    my_grad = my_model.backward(my_model(x), y, x)
    print("my grad: ", my_grad)

    optimizer.step()
    print("torch update w: ", torch_model.linear.weight)

    # my_new_w = my_sgd(lr, npy_w, my_grad)
    my_new_w = my_adam(lr, npy_w, my_grad)
    print("my new w: ", my_new_w)
