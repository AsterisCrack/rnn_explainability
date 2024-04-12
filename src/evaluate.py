import torch

from .train_functions import t_step

from .utils import load_model, set_seed

from .data import load_data

from typing import Final


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 2222
set_seed(SEED)

DATA_PATH: Final[str] = "./NLP_Data/data"


def main() -> None:
    # load model
    print("Loading model...")
    model = load_model("best_model")

    # load the data
    print("Loading data...")
    train_dataloader, val_dataloader, test_dataloader = load_data(
        save_path=DATA_PATH)

    # perform testing
    print("Testing model...")
    test_accuracy = t_step(model, test_dataloader, device=DEVICE)

    print(f"Test accuracy: {test_accuracy:.4f}")

    return


if __name__ == "__main__":
    main()
