import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from loader import load_data
from model import BertModule
from evaluator import Evaluator


def main():
    config = Config
    model = BertModule(config)
    print(model)
    # load data
    train_data_loader,valid_data_loader = get_date_loader(config)
    # adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=Config["learning_rate"])
    # evaluator
    evaluator = Evaluator(config, model, valid_data_loader)
    # timer
    a_start = time.time()

    for epoch in range(config["epoch"]):
        print(f"epoch {epoch+1} begin")
        p_start = time.time()

        # start training
        model.train()
        loss_list = []
        for sentence, positive_sentence, negative_sentence in train_data_loader:
            # empty gradient
            optimizer.zero_grad()
            # loss
            loss = model(sentence, positive_sentence, negative_sentence)
            # back propagation
            loss.backward()
            # update weights
            optimizer.step()

            loss_list.append(loss.item())
        p_end = time.time()
        print(f"epoch [{epoch+1}] used time: {p_end - p_start}, avg loss: {np.mean(loss_list)}")

        # evaluate
        evaluator.eval(epoch+1)

    a_end = time.time()
    print(f"total used time: {a_end - a_start}")
    torch.save(model.state_dict(), "model.pth")


def get_date_loader(config):
    train_data, valid_data = load_data(config)
    return DataLoader(train_data, shuffle=True, batch_size=Config["batch_size"]),DataLoader(valid_data, shuffle=True, batch_size=Config["batch_size"])


if __name__ == '__main__':
    main()