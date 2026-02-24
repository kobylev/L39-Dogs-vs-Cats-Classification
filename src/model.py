"""
Define the CNN model for binary classification (cats vs dogs).
"""

import torch
import torch.nn as nn


class CatDogCNN(nn.Module):
    """
    A custom CNN for binary classification of cats vs dogs.
    
    Architecture:
    - Input: RGB images of size 150x150
    - 3 convolutional blocks with increasing channels
    - Batch normalization and max pooling in each block
    - Fully connected layers for classification
    - Output: single logit for binary cross-entropy with logits
    """
    
    def __init__(self):
        super(CatDogCNN, self).__init__()
        
        # Block 1: Conv -> BN -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2: Conv -> BN -> ReLU -> MaxPool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3: Conv -> BN -> ReLU -> MaxPool
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # After 3 pooling layers: 150 -> 75 -> 37 -> 18
        # Plus padding effects, we get approximately 18x18 spatial dims
        self.fc1 = nn.Linear(128 * 18 * 18, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)  # Binary output
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)
        
        # Flatten and fully connected
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
