# deep learning libraries
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from typing import Optional


@torch.enable_grad()
def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    mean: float,
    std: float,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function train the model.

    Args:
        model: model to train.
        train_data: dataloader of train data.
        mean: mean of the target.
        std: std of the target.
        loss: loss function.
        optimizer: optimizer.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.
    """

    # TODO
    # this function performs one step of the training loop

    model.train()  # set the model to training mode

    model.to(device)  # move the model to the device

    losses = []  # initialise the list of losses

    for x, y in train_data:
        x, y = x.to(device), y.to(device)  # move the data to the device

        optimizer.zero_grad()  # zero the gradients

        y_hat = model(x)  # forward pass

        y_hat = y_hat.to(device)  # move the prediction to the device

        y_hat = y_hat * std + mean  # rescale the prediction
        y = y * std + mean  # rescale the target

        loss_iter = loss(y_hat, y)  # calculate the loss

        loss_iter.backward()  # backpropagation

        optimizer.step()  # update the weights

        losses.append(loss_iter.item())  # store the loss

    print(f"\tAverage train loss epoch {epoch}: ", np.mean(losses))
    writer.add_scalar("Loss/train", np.mean(losses), epoch)  # log the loss

    return None


@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    mean: float,
    std: float,
    loss: torch.nn.Module,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function train the model.

    Args:
        model: model to train.
        val_data: dataloader of validation data.
        mean: mean of the target.
        std: std of the target.
        loss: loss function.
        scheduler: scheduler.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.
    """

    # TODO

    # this function performs one step of the validation loop

    model.eval()  # set the model to evaluation mode

    losses = []  # initialise the list of losses

    for x, y in val_data:
        x, y = x.to(device), y.to(device)  # move the data to the device

        y_hat = model(x)  # forward pass

        y_hat = y_hat * std + mean  # rescale the prediction

        y = y * std + mean  # rescale the target

        loss_iter = loss(y_hat, y)  # calculate the loss

        losses.append(loss_iter.item())  # store the loss

    print(f"\tAverage val loss epoch {epoch}: ", np.mean(losses))
    writer.add_scalar("Loss/val", np.mean(losses), epoch)  # log the loss

    if scheduler is not None:  # if a scheduler is provided
        scheduler.step(np.mean(losses))  # update the learning rate

    return None


@torch.no_grad()
def t_step(
    model: torch.nn.Module,
    test_data: DataLoader,
    mean: float,
    std: float,
    device: torch.device,
) -> float:
    """
    This function tests the model.

    Args:
        model: model to make predcitions.
        test_data: dataset for testing.
        mean: mean of the target.
        std: std of the target.
        device: device for running operations.

    Returns:
        mae of the test data.
    """

    # TODO

    # this function tests the model

    model.eval()  # set the model to evaluation mode
    model.to(device)  # move the model to the device

    loss_fn = torch.nn.L1Loss()

    losses = []  # initialise the list of losses

    for x, y in test_data:
        x, y = x.to(device), y.to(device)  # move the data to the device

        y_hat = model(x)

        y_hat = y_hat * std + mean
        y = y * std + mean

        loss_iter = loss_fn(y_hat, y)

        losses.append(loss_iter.item())

    return np.mean(losses)
