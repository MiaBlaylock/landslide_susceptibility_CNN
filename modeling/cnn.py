import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, input_channels=4):
        """
        A simple CNN for pixel-wise classification, outputting a heatmap of probabilities.

        Args:
            input_channels (int): Number of input channels (e.g., 4 for multi-band data).
        """
        super(CNNModel, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2x

        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2x

        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2x

        # Upsampling layers
        self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # Upsample by 2x
        self.upsample2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # Upsample by 2x
        self.upsample3 = nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)  # Upsample by 2x

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1, H, W).
        """
        # Downsampling
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Upsampling
        x = F.relu(self.upsample1(x))
        x = F.relu(self.upsample2(x))
        x = torch.sigmoid(self.upsample3(x))  # Sigmoid activation for probabilities

        return x