# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.jit import RecursiveScriptModule

# other libraries
from typing import Final

# own modules
from src.RNNModelTrain.data import load_data
from src.RNNModelTrain.utils import set_seed
from src.RNNModelTrain.train_functions import t_step

# static variables
DATA_PATH: Final[str] = "data"
NUM_CLASSES: Final[int] = 10

# set device
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def main() -> None:
    """
    This function is the main program.
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

    # load the model
    model: RecursiveScriptModule = torch.jit.load("models/best_model.pt")

    # TODO: print the parameters of the model, and the shape of the fc layer

    # evaluate the model
    test_results = t_step(model, test_dataloader,
                          mean_price, std_price, device)

    # print the test resulsts
    print(f"MAE test loss: {test_results}")

    return None


if __name__ == "__main__":
    main()
