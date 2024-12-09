import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class LandslideDataset(Dataset):
    def __init__(self, image_files, label_files, transform=None):
        """
        Dataset for loading tiles and their labels.
        Args:
            image_files (list): List of file paths to image tiles.
            label_files (list): List of file paths to label tiles.
            transform (callable, optional): Transform to apply to images.
        """
        self.image_files = image_files  # Use the list of file paths directly
        self.label_files = label_files
        self.transform = transform

        assert len(self.image_files) == len(self.label_files), "Mismatch in number of images and labels!"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and label
        image = np.load(self.image_files[idx])
        label = np.load(self.label_files[idx])
        #print(f"Raw label shape: {label.shape}")  # Add this for debugging
        #analyze_channels(label) #add for debugging
        # Check for NaN values in the image
        # if np.isnan(image).any():
        #     print(f"NaN found in image file: {self.image_files[idx]}")

        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)


        # Convert to PyTorch tensors
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)  # Add channel dimension for label

        # Apply transformations (e.g., normalization)
        if self.transform:
            image = self.transform(image)

        return image, label

# def analyze_channels(label):

#     print(f"Label shape: {label.shape}")
#     for i in range(label.shape[0]):
#         unique_values = np.unique(label[i, :, :])
#         print(f"Channel {i+1} unique values: {unique_values}")
#         print(f"Channel {i+1}: Min={label[i, :, :].min()}, Max={label[i, :, :].max()}, Mean={label[i, :, :].mean()}")
#         print()

#     # Visualize channels
#     fig, axes = plt.subplots(1, label.shape[0], figsize=(15, 5))
#     for i in range(label.shape[0]):
#         axes[i].imshow(label[i, :, :], cmap="gray")
#         axes[i].set_title(f"Channel {i+1}")
#         axes[i].axis("off")
#     plt.tight_layout()
#     plt.show()

def get_dataloaders(image_dir, label_dir, batch_size=32, test_split=0.15, val_split=0.15, shuffle=True):
    """
    Create DataLoaders for training, validation, and testing.

    Args:
        image_dir (str): Path to the directory containing image tiles.
        label_dir (str): Path to the directory containing label tiles.
        batch_size (int): Batch size for DataLoader.
        test_split (float): Fraction of data to use for testing.
        val_split (float): Fraction of data to use for validation.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        tuple: Training, validation, and testing DataLoaders.
    """
    # Load all file paths
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.npy')])
    label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.npy')])

    # Split into train, validation, and test sets
    train_images, test_images, train_labels, test_labels = train_test_split(image_files, label_files, test_size=test_split, random_state=42)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=val_split, random_state=42)

    # Create PyTorch datasets
    train_dataset = LandslideDataset(train_images, train_labels)
    val_dataset = LandslideDataset(val_images, val_labels)
    test_dataset = LandslideDataset(test_images, test_labels)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
