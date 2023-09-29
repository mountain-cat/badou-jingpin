import torch
import numpy as np


def to_one_hot(x):
    lens = len(x)
    max_class = np.max(x) + 1
    one_hot = np.zeros((lens, max_class))
    for i in range(lens):
        one_hot[i, x[i]] = 1

    return one_hot


def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)  # 按轴1方向计算softmax


def ce(y_pred, y_true):
    y_true = to_one_hot(y_true)
    y_pred = softmax(y_pred)
    return np.mean(-1 * np.sum(y_true * np.log(y_pred), axis=1))  # 先按轴1计算cross entropy， 然后按轴0计算均值


if __name__ == '__main__':
    gt_arr = np.array([1, 3, 2], dtype="long")
    pred_arr = np.array([
        [0.3, 0.6, 0.2, 0.5],
        [0.5, 0.4, 0.7, 0.3],
        [0.6, 0.2, 0.4, 0.1]
    ], dtype="f4")
    print(ce(pred_arr, gt_arr))

    gt_tensor = torch.LongTensor(gt_arr)  # torch 整型输入类型必须为long
    pred_tensor = torch.FloatTensor(pred_arr)

    ce_loss = torch.nn.CrossEntropyLoss()
    print(ce_loss(pred_tensor, gt_tensor).numpy())
