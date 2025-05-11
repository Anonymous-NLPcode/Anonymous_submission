import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output1, output2):
        distance = torch.dist(output1, output2, p=2)  # 这里使用欧氏距离
        return distance
