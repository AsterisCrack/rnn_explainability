# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
# from tqdm.auto import tqdm
from typing import Final

# own modules
from src.RNNModelTrain.data import load_data
from src.RNNModelTrain.models import RNNModel
from src.RNNModelTrain.train_functions import train_step, val_step
from src.RNNModelTrain.utils import set_seed, save_model

# static variables
DATA_PATH: Final[str] = "data"

# set device and seed
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"Using Device: {device}")

set_seed(42)

# HYPERPARAMETER INITIALIZATION
HIDDEN_SIZE: int = 32
LEARNING_RATE: float = 7.5e-4
EPOCHS: int = 200
BATCH_SIZE: int = 64
PATIENCE: int = 40
LR_DECREASE_FACTOR: float = 0.1  # new_lr = lr * factor


def main() -> None:
    """
    This function is the main program for training.
    """

    # TODO

    # load the data
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    test_dataloader: DataLoader
    mean_price: float
    std_price: float
    (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        mean_price,
        std_price
    ) = load_data(save_path=DATA_PATH)

    # initialize the model
    model = RNNModel(HIDDEN_SIZE).to(device)

    # initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # initialize the loss function (MAE)
    loss_fn = torch.nn.L1Loss()

    # initialize the summary writer
    writer = SummaryWriter()

    # initialise the scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=PATIENCE, factor=LR_DECREASE_FACTOR, verbose=True
    )

    # train the model
    for epoch in range(EPOCHS):
        print(f"TRAINING: Epoch {epoch} / {EPOCHS}")

        # train the model
        train_step(model,
                   train_dataloader,
                   mean_price,
                   std_price,
                   loss_fn,
                   optimizer,
                   writer,
                   epoch,
                   device)

        # validate the model
        val_step(model,
                 val_dataloader,
                 mean_price,
                 std_price,
                 loss_fn,
                 scheduler,
                 writer,
                 epoch,
                 device)

    print("Saving model...")
    # save the model
    # delete ./models/best_model.pt to save the new model otherwise it doesnt update it)
    try:
        import os
        os.remove("models/best_model.pt")
    except Exception:
        pass

    save_model(model, "best_model")
    save_model(model, f"hidden_{HIDDEN_SIZE}_lr_{LEARNING_RATE}_epochs_{EPOCHS}" +
               f"_batch_{BATCH_SIZE}_patience_{PATIENCE}")

    return None


if __name__ == "__main__":
    main()
