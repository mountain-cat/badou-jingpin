import numpy as np
import matplotlib.pyplot as plt


def func(x, w):
    w1, w2, w3 = w
    return w1 * x ** 2 + w2 * x + w3


def loss(y_pred, y_true):
    return (y_pred - y_true) ** 2


if __name__ == "__main__":
    
    np.random.seed(2023)
    X = np.random.randn(100, 1)
    Y = np.array([3 * x ** 2 + 5 * x + 8 for x in X])


    w1, w2, w3 = 0.1, 0.2, 0.3
    lr = 0.03
    epoch = 1000

    for e in range(epoch):
        e_loss = 0
        for x, y in zip(X, Y):
            y_pred = func(x, [w1, w2, w3])
            e_loss += loss(y_pred, y)
            gw1 = 2 * (y_pred - y) * x ** 2
            gw2 = 2 * (y_pred - y) * x
            gw3 = 2 * (y_pred - y)
            w1 -= lr * gw1
            w2 -= lr * gw2
            w3 -= lr * gw3
        
        e_loss /= len(x)
        print("Epoch: %d, loss: %.6f" % (e, e_loss))
        if e_loss < 1e-6:
            break

    print("final w1: %.2f, w2: %.2f, w3: %.2f" % (w1, w2, w3))

    y_pred = [func(x, [w1, w2, w3]) for x in X]

    plt.figure()
    plt.scatter(X, Y, c="r", marker="o")
    plt.scatter(X, y_pred, c="k", s=1)
    plt.tight_layout()
    plt.show()
