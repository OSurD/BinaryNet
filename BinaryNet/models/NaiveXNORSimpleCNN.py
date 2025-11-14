import torch
from BinaryNet.binary_layers.NaiveXNOR import NaiveXNORConv2d, NaiveXNORLinear

class NaiveXNORSimpleCNN(torch.nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, (5,5), padding=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2), stride=2),
            torch.nn.BatchNorm2d(64),
            NaiveXNORConv2d(64, 96, 3, padding=1),
            torch.nn.MaxPool2d((2,2), stride=2),
            torch.nn.BatchNorm2d(96),
            NaiveXNORConv2d(96, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            NaiveXNORConv2d(128, 256, 3, padding=1),
            torch.nn.AdaptiveAvgPool2d((1,1)),
            torch.nn.Flatten()
        )
        self.fc = torch.nn.Sequential(
            NaiveXNORLinear(256, 64),
            torch.nn.Linear(64, n_classes)
        )
        
    def forward(self, x):
        z = self.net(x)
        return self.fc(z)
