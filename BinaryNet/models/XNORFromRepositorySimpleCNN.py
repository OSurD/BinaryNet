import torch
from BinaryNet.binary_layers.XNORFromRepository import BinConv2d

class XNORFromRepositorySimpleCNN(torch.nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, (5,5), padding=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2), stride=2),
            BinConv2d(64, 96, (3,3), padding=1),
            torch.nn.MaxPool2d((2,2), stride=2),
            BinConv2d(96, 128, (3,3), padding=1),
            BinConv2d(128, 256, (3,3), padding=1),
            torch.nn.AdaptiveAvgPool2d((1,1)),
            torch.nn.Flatten()
        )
        self.fc = torch.nn.Sequential(
            BinConv2d(256, 64, Linear=True),
            torch.nn.Linear(64, n_classes)
        )
        
    def forward(self, x):
        z = self.net(x)
        return self.fc(z)