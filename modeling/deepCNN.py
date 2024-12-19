import torch
import torch.nn as nn


class DeepCNNModel(nn.Module):
    def __init__(self, input_channels=4, output_channels=1):
        """
        A deep CNN for pixel-wise classification (e.g., segmentation).

        Args:
            input_channels (int): Number of input channels (e.g., 4 for multi-band data).
            output_channels (int): Number of output channels (e.g., 1 for binary segmentation).
        """
        super(DeepCNNModel, self).__init__()

        # Downsampling blocks
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample (H/2, W/2)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample (H/4, W/4)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample (H/8, W/8)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample (H/16, W/16)

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample (H/32, W/32)
        )

        # Upsampling blocks
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # Upsample (H/16, W/16)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # Upsample (H/8, W/8)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # Upsample (H/4, W/4)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # Upsample (H/2, W/2)
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=2, stride=2),  # Upsample to (H, W)
            nn.Sigmoid()  # Output probabilities
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x