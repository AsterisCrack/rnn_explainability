# import the necessary libraries
import torch
from torch.jit import RecursiveScriptModule
from torch.utils.data import DataLoader

import numpy as np
import random
import os
import matplotlib.pyplot as plt


@torch.no_grad()
def parameters_to_double(model: torch.nn.Module) -> None:
    """
    This function transforms the model parameters to double.

    Args:
        model: pytorch model.
    """

    # TODO
    model.double()


def save_model(model: torch.nn.Module, name: str) -> None:
    """
    This function saves a model in the 'models' folder as a torch.jit.
    It should create the 'models' if it doesn't already exist.

    Args:
        model: pytorch model.
        name: name of the model (without the extension, e.g. name.pt).
    """

    # create folder if it does not exist
    if not os.path.isdir("models"):
        os.makedirs("models")

    # save scripted model
    model_scripted: RecursiveScriptModule = torch.jit.script(model.cpu())
    model_scripted.save(f"models/{name}.pt")

    return None


def load_model(name: str) -> RecursiveScriptModule:
    """
    This function is to load a model from the 'models' folder.

    Args:
        name: name of the model to load.

    Returns:
        model in torchscript.
    """

    # define model
    model: RecursiveScriptModule = torch.jit.load(f"models/{name}.pt")

    return model


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior.

    Args:
        seed: seed number to fix radomness.
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None


def calculate_accuracy(model: torch.nn.Module, dataloader: DataLoader, threshold: float = 0.5, device: str = 'cpu') -> float:
    """
    Calculate the accuracy of a PyTorch model given a DataLoader.

    The function moves the model to the specified device, sets it to evaluation mode, and computes
    the accuracy by comparing the model's predictions against the true labels. The predictions are
    determined based on a specified threshold.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        dataloader (DataLoader): The DataLoader containing the dataset to evaluate against.
        threshold (float, optional): Probability threshold to predict a sample as positive. Defaults to 0.5.
        device (str, optional): Device to which the model and data are moved ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        float: The accuracy of the model on the given dataset.
    """
    # TODO: Calculate accuracy of a model given a dataloader
    accuracy = None

    # move the model to the device
    model.to(device)

    # set the model to evaluation mode
    model.eval()

    # initialise the number of correct predictions
    correct = 0

    # iterate through the dataloader
    for texts, labels, lengths in dataloader:
        texts, labels = texts.to(device), labels.to(device)

        # get the model's predictions
        predictions = model(texts, lengths)

        # convert the predictions to binary
        predictions = (predictions > threshold).float()

        # calculate the number of correct predictions
        correct += (predictions == labels).sum().item()

    # calculate the accuracy
    accuracy = correct / len(dataloader.dataset)

    return accuracy


def plot_accuracies(model: torch.nn.Module,
                    train_dataloader: DataLoader,
                    val_dataloader: DataLoader,
                    test_dataloader: DataLoader,
                    train_accuracies: dict,
                    val_accuracies: dict,
                    model_name: str,
                    device: str = "cpu") -> None:
    """
    Plots the accuracies of the model on the training, validation, and test sets.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        train_dataloader (DataLoader): The DataLoader containing the training dataset.
        val_dataloader (DataLoader): The DataLoader containing the validation dataset.
        test_dataloader (DataLoader): The DataLoader containing the test dataset.
        device (str): Device to which the model and data are moved ('cpu' or 'cuda').
    """

    # get the accuracies for train, validation and test sets
    last_train_accuracy = calculate_accuracy(
        model, train_dataloader, device=device)
    last_val_accuracy = calculate_accuracy(
        model, val_dataloader, device=device)
    test_accuracy = calculate_accuracy(model, test_dataloader, device=device)

    # print the accuracies
    print(f"Train Accuracy: {last_train_accuracy:.4f}")
    print(f"Validation Accuracy: {last_val_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Obtain epochs where accuracy was calculated in order to plot them
    num_epochs, train_accuracies = zip(
        *sorted(train_accuracies.items()))
    _, val_accuracies = zip(*sorted(val_accuracies.items()))

    # Plot the evolution during training
    plt.plot(num_epochs, train_accuracies,
             label='RNN Train', linestyle='-', color='blue')
    plt.plot(num_epochs, val_accuracies,
             label='RNN Validation', linestyle='--', color='blue')
    plt.axhline(y=test_accuracy, label='RNN Test',
                linestyle='-.', color='lightblue', alpha=0.5)
    plt.suptitle('Recurrent Neural Network model Accuracy Evolution')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # save the plot into the figures/ folder
    plt.savefig(f"figures/accuracy_evolution_{model_name}.png")
    plt.show()
