import torch
import torch.nn as nn

class mClassifier(nn.Module):
    def __init__(self):
        super(mClassifier, self).__init__()
        self.fc = nn.Linear(6, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))
