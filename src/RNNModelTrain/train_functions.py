# import necessary dependencies
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# import the writer
from torch.utils.tensorboard import SummaryWriter

from typing import Dict, Tuple

from src.RNNModelTrain.utils import calculate_accuracy


def train_torch_model(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epochs: int,
    print_every: int,
    patience: int,
    writer: SummaryWriter,
    device: str = 'cpu'
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Train and validate the logistic regression model.

    Args:
        model (torch.nn.Module): An instance of the model to be trained.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        val_dataloader (DataLoader): DataLoader for the validation dataset.
        learning_rate (float): The learning rate for the optimizer.
        criterion (nn.Module): Loss function to use for training.
        optimizer (optim.Optimizer): Optimizer to use for training.
        epochs (int): The number of epochs to train the model.
        print_every (int): Frequency of epochs to print training and validation loss.
        patience (int): The number of epochs to wait for improvement on the validation loss before stopping training early.
        device (str): device where to train the model.

    Returns:
        Tuple[Dict[int, float],Dict[int, float]]: Dictionary of accuracies at each `print_every` interval for the training and validation datasets.
    """
    # TODO: Initialize dictionaries to store training and validation accuracies
    train_accuracies: Dict[int, float] = {}  # epoch: accuracy
    val_accuracies: Dict[int, float] = {}  # epoch: accuracy

    # TODO: Initialize variables for Early Stopping
    best_loss: float = float('inf')
    epochs_no_improve: int = 0

    # TODO: Move the model to the specified device (CPU or GPU)
    model.to(device)

    # TODO: Implement the training loop over the specified number of epochs
    for epoch in range(epochs):
        # TODO: Set the model to training mode
        model.train()

        total_loss: float = 0.0

        # TODO: Implement the loop for training over each batch in the training dataloader
        for features, labels, text_len in train_dataloader:

            # TODO: Move features and labels to the specified device
            features, labels = features.to(device), labels.to(device)

            # TODO: Clear the gradients
            optimizer.zero_grad()

            # TODO: Forward pass (compute the model output)
            output = model(features, text_len)

            # TODO: Compute the loss
            # cast labels to float to avoid a data type mismatch error
            loss = criterion(output, labels.float())

            # TODO: Backward pass (compute the gradients)
            loss.backward()

            # TODO: Update model parameters
            optimizer.step()

            # TODO: Accumulate the loss
            total_loss += loss.item()

        # TODO: Implement the evaluation phase
        model.eval()
        val_loss: float = 0.0

        with torch.no_grad():
            # TODO: Loop over the validation dataloader
            for features, labels, text_len in val_dataloader:

                # TODO: Move features and labels to the specified device
                features, labels = features.to(device), labels.to(device)

                # TODO: Forward pass (compute the model output)
                output = model(features, text_len)

                # TODO: Compute the loss
                loss = criterion(output, labels.float())

                # TODO: Accumulate validation loss
                val_loss += loss.item()

        # TODO: Calculate training and validation accuracy
        train_accuracy = calculate_accuracy(
            model, train_dataloader, device=device)
        val_accuracy = calculate_accuracy(
            model, val_dataloader, device=device)

        # TODO: Store accuracies
        train_accuracies[epoch] = train_accuracy
        val_accuracies[epoch] = val_accuracy

        # write the training loss to tensorboard
        writer.add_scalar("Loss/train", total_loss /
                          len(train_dataloader), epoch)
        writer.add_scalar("Loss/val", val_loss / len(val_dataloader), epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)

        # TODO: Print training and validation results every 'print_every' epochs
        if epoch % print_every == 0 or epoch == epochs - 1:
            # TODO: Calculate and print average losses and accuracies
            avg_train_loss = total_loss / len(train_dataloader)
            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Epoch {epoch} -> \
                    Train Loss: {avg_train_loss:.4f}, \
                    Val Loss: {avg_val_loss:.4f}, \
                    Train Acc: {train_accuracy:.4f}, \
                    Val Acc: {val_accuracy:.4f}")

        # TODO: Implement Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return train_accuracies, val_accuracies
