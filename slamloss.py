import torch
import torch.nn as nn
import numpy as np


class SlamLoss(nn.Module):
    def __init__(self):
        super(SlamLoss, self).__init__()

    def forward(self, pred, truth):
