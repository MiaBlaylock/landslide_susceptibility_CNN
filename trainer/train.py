import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataloader.data_loader import get_dataloaders
from modeling.cnn import CNNModel

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=10,
    validate_every=1,
    device='cuda',
    checkpoint_dir="models/checkpoints",
    log_dir="runs/landslide_cnn",
):
    """
    Trains the CNN model and logs metrics.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        num_epochs (int): Number of epochs to train.
        device (str): Device to use ('cuda' or 'cpu').
        checkpoint_dir (str): Directory to save model checkpoints.
        log_dir (str): Directory for TensorBoard logs.

    Returns:
        dict: Dictionary containing training and validation loss history.
    """
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    # Move model to device
    model = model.to(device)

    # Training and validation loss history
    history = {"train_loss": [], "val_loss": []}

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move data to device
            images, labels = images.to(device), labels.to(device)

            # print(f"Images: min={images.min().item()}, max={images.max().item()}")
            # print(f"Labels: min={labels.min().item()}, max={labels.max().item()}")

            if torch.isnan(images).any() or torch.isinf(images).any():
                raise ValueError("Input images contain NaN or Inf!")
            if torch.isnan(labels).any() or torch.isinf(labels).any():
                raise ValueError("Labels contain NaN or Inf!")

            # Forward pass
            outputs = model(images)  # Output shape: (batch_size, 1, H, W)
            # print(f"Output shape: {outputs.shape}")
            # print(f"Label shape: {labels.shape}")
            # print(f"Output range: min={outputs.min().item()}, max={outputs.max().item()}")

            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

        # Log training loss
        avg_train_loss = running_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")
        writer.add_scalar('Loss/Train', avg_train_loss, epoch+1)


        if (epoch + 1) % validate_every == 0:
            # Validate the model
            val_loss = validate_model(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")
            writer.add_scalar('Loss/Validation', val_loss, epoch+1)

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

    writer.close()
    return history

def validate_model(model, val_loader, criterion, device='cuda'):
    """
    Validates the model on the validation dataset.

    Args:
        model (torch.nn.Module): The model to validate.
        val_loader (DataLoader): Validation data loader.
        criterion (torch.nn.Module): Loss function.
        device (str): Device to use ('cuda' or 'cpu').

    Returns:
        float: Average validation loss.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            # Move data to device
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Accumulate loss
            running_loss += loss.item()

    avg_val_loss = running_loss / len(val_loader)
    return avg_val_loss
