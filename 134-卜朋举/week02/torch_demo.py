import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class BinaryClassModel(nn.Module):

    def __init__(self, input_dim):
        super(BinaryClassModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss()

    def forward(self, x, y=None):
        x = self.linear(x)
        x = self.sigmoid(x)

        if y is not None:
            return self.loss(x, y)
        else:
            return x


def build_sample():
    x = np.random.rand(5)

    if x[0] > x[-1]:
        return x, 1
    else:
        return x, 0


def build_dataset(n_sample):
    x = []
    y = []
    np.random.seed(2023)
    for _ in range(n_sample):
        xi, yi = build_sample()
        x.append(xi)
        y.append([yi])

    return torch.FloatTensor(np.array(x)), torch.FloatTensor(np.array(y))


def evaluate(model):
    model.eval()
    n_sample = 100
    x, y = build_dataset(n_sample)
    print("positive count: %d, negative count: %d" % (sum(y.numpy()), sum(1 - y.numpy())))
    right, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_hat, y_true in zip(y_pred, y):
            if y_hat > 0.5 and y_true == 1:
                right += 1
            elif y_hat <= 0.5 and y_true == 0:
                right += 1
            else:
                wrong += 1
    print("num sample: %d, right: %d, wrong: %d, acc: %f" % (n_sample, right, wrong, right / (right + wrong)))
    return right / (right + wrong)


def predict(model_path, x):
    input_dim = 5
    model = BinaryClassModel(input_dim)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    with torch.no_grad():
        pred = model.forward(torch.FloatTensor(x))

    for xi, p in zip(x, pred.numpy()):
        print("input : %s, pred_cate: %d, p: %f" % (xi, np.round(p), p))


def main():
    epoch = 10
    batch_size = 20
    n_sample_train = 5000
    lr = 0.01
    input_size = 5

    model = BinaryClassModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    x_train, y_train = build_dataset(n_sample_train)
    train_log = []
    for e in range(epoch):
        model.train()
        watch_loss = []
        for i in range(n_sample_train // batch_size):
            x = x_train[batch_size * i: (i + 1) * batch_size]
            y = y_train[batch_size * i: (i + 1) * batch_size]
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())

        print("epoch: %d, loss: %f" % (e + 1, float(np.mean(watch_loss))))
        acc = evaluate(model)
        train_log.append([acc, float(np.mean(watch_loss))])

    torch.save(model.state_dict(), "./out/model.pth")

    plt.figure()
    plt.plot(range(len(train_log)), [log[0] for log in train_log], label="acc", c="red")
    plt.plot(range(len(train_log)), [log[1] for log in train_log], label="loss", c="blue")
    plt.show()


if __name__ == '__main__':
    main()

    test_vec = np.random.random(15).reshape(3, 5)
    predict("./out/model.pth", test_vec)
