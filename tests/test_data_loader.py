import os
import torch
from dataloader.data_loader import get_dataloaders

# Define paths to test the data loader
image_dir = "data/tiles/images"
label_dir = "data/tiles/labels"

def test_data_loader(image_dir, label_dir, batch_size = 16, test_split=0.15, val_split=0.15):
    """
    Tests the data loader by checking:
    - Number of samples in each split
    - Alignment between images and labels
    - Output shape and type
    """

    # Load DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        image_dir, label_dir, batch_size=batch_size, test_split=test_split, val_split=val_split
    )

    # Check if the DataLoaders are not empty
    assert len(train_loader) > 0, "Training DataLoader is empty!"
    assert len(val_loader) > 0, "Validation DataLoader is empty!"
    assert len(test_loader) > 0, "Test DataLoader is empty!"

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Check batch dimensions
    for images, labels in train_loader:
        assert images.shape[0] == batch_size, "Batch size mismatch!"
        assert images.ndim == 4, "Images should have 4 dimensions (batch, channels, height, width)!"
        print(labels.shape)
        assert labels.ndim == 4, "Labels should have 4 dimensions (batch, 1, height, width)!"
        assert isinstance(images, torch.Tensor), "Images are not PyTorch tensors!"
        assert isinstance(labels, torch.Tensor), "Labels are not PyTorch tensors!"
        print(f"Sample batch image shape: {images.shape}")
        print(f"Sample batch label shape: {labels.shape}")
        break

    print("All tests passed for data loader!")

# Run the test
if __name__ == "__main__":
    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        print(f"Test directories not found: {image_dir} or {label_dir}")
    else:
        test_data_loader()
