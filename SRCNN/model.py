import torch
import torch.nn as nn
import torch.nn.functional as F

class SRCNN:
    def __init__(self):
        super().__init__()
        self.SRCNN = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=2)
        )
    
    def forward(self, x):
        x = self.SRCNN(x)
        return x
    



