import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import torchvision.transforms as T
from torchvision.transforms.functional import rotate
from sklearn.model_selection import train_test_split

class LandslideDataset(Dataset):
    def __init__(self, image_files, label_files, mean=None, std=None, augment=False, compute_norm=False):
        """
        Dataset for loading tiles and their labels.
        Args:
            image_files (list): List of file paths to image tiles.
            label_files (list): List of file paths to label tiles.
            mean (list): Precomputed mean for normalization.
            std (list): Precomputed std for normalization.
            augment (bool): Whether to apply augmentations.
            compute_norm (bool): Compute mean and std dynamically for normalization.
        """
        self.image_files = image_files
        self.label_files = label_files
        self.augment = augment

        # Compute normalization parameters if required
        if compute_norm:
            self.mean, self.std = self.compute_normalization()
            print(f"Computed mean: {self.mean}, std: {self.std}")
        else:
            self.mean = mean if mean else [0.0] * 4  # Default to zeros if not provided
            self.std = std if std else [1.0] * 4    # Default to ones if not provided

        # Define normalization
        self.normalize = T.Normalize(mean=self.mean, std=self.std)

    def compute_normalization(self):
        """Compute mean and std across all images and bands."""
        all_sums = np.zeros(4)  # Assuming 4 bands
        all_squares = np.zeros(4)
        pixel_count = 0

        for file in self.image_files:
            image = np.load(file)  # Shape: (bands, H, W)
            image = np.nan_to_num(image, nan=0.0)  # Replace NaNs
            all_sums += image.sum(axis=(1, 2))
            all_squares += (image ** 2).sum(axis=(1, 2))
            pixel_count += image.shape[1] * image.shape[2]

        mean = all_sums / pixel_count
        std = np.sqrt(all_squares / pixel_count - mean ** 2)
        return mean.tolist(), std.tolist()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and label
        image = np.load(self.image_files[idx])  # (bands, H, W)
        label = np.load(self.label_files[idx])  # (H, W)

        # Replace NaN values
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        # Convert to tensor
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        # Apply augmentations
        if self.augment:
            image, label = self.apply_augmentation(image, label)

        # Normalize
        image = self.normalize(image)

        return image, label

    def apply_augmentation(self, image, label):
        """Apply geometric augmentations to the image and label."""
        # Random horizontal flip
        if random.random() > 0.5:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[2])

        # Random vertical flip
        if random.random() > 0.5:
            image = torch.flip(image, dims=[1])
            label = torch.flip(label, dims=[1])

        # Random rotation
        angle = random.uniform(-10, 10)
        image = rotate(image, angle)
        label = rotate(label, angle)

        # Random affine transformations
        affine_transform = T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-5, 5))
        image = affine_transform(image)
        label = affine_transform(label)

        return image, label


def get_dataloaders(image_dir, label_dir, batch_size=32, test_split=0.15, val_split=0.15, shuffle=True):
    # Load all file paths
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.npy')])
    label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.npy')])

    # Split into train, validation, and test sets
    train_images, test_images, train_labels, test_labels = train_test_split(image_files, label_files, test_size=test_split, random_state=42)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=val_split, random_state=42)

    # Compute normalization statistics only on the training set
    train_dataset_for_norm = LandslideDataset(train_images, train_labels, compute_norm=True)
    mean, std = train_dataset_for_norm.mean, train_dataset_for_norm.std

    # Create datasets
    train_dataset = LandslideDataset(train_images, train_labels, mean=mean, std=std, augment=True)
    val_dataset = LandslideDataset(val_images, val_labels, mean=mean, std=std, augment=False)
    test_dataset = LandslideDataset(test_images, test_labels, mean=mean, std=std, augment=False)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
