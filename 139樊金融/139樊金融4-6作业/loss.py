import torch
import torch.nn as nn
from torchcrf import CRF


class loss(nn.Module):

    def __init__(self, gamma=2, eps=1e-7):
        super(loss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = CRF(4, batch_first=True)

    def forward(self, input, target):
        logp = -self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()