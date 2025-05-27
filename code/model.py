import torch.nn as nn
class MLE(nn.Module):
    def __init__(self):
        super(MLE, self).__init__()
        self.layer = nn.Linear(2,5)

    def forward(self, x):
        out = self.layer(x)
        return out