import torch
from torch import nn


class NeuralNetwork(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(in_features=64 * 28 * 28, out_features=2)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)