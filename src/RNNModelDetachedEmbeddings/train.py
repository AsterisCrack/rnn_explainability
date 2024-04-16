import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from typing import Final, Dict

from src.RNNModelDetachedEmbeddings.utils import set_seed, save_model, plot_accuracies
from src.RNNModelDetachedEmbeddings.models import RNNModel
from src.RNNModelDetachedEmbeddings.data import load_data
from src.RNNModelDetachedEmbeddings.train_functions import train_torch_model

from gensim.models.keyedvectors import load_word2vec_format

SEED: Final[int] = 2222
set_seed(SEED)

DATA_PATH: Final[str] = "./NLP_Data/data"

batch_size: int = 16
epochs: int = 3
print_every: int = 1
patience: int = 5
learning_rate: float = 1e-4
hidden_dim: int = 256
num_layers: int = 1

dataset_name: str = "IMDB"  # "TweepFake"  #

model_name = f"{dataset_name}_DE_rnn_hidden_{hidden_dim}_lr_{learning_rate}_epochs_{epochs}" + \
    f"_batch_{batch_size}_patience_{patience}"

# TODO: Check if GPU is available and move the model to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on {device}")
print(f"Training on dataset: {dataset_name}")


def main() -> None:
    """
    This function is the main program for training.
    """

    print("Loading data...")
    # load the data
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    test_dataloader: DataLoader
    (
        train_dataloader,
        val_dataloader,
        test_dataloader
    ) = load_data(save_path=DATA_PATH, dataset_name=dataset_name, batch_size=batch_size)

    print("Loading embeddings...")
    w2v_model = load_word2vec_format(
        "./NLP_Data/embeddings/GoogleNews-vectors-negative300.bin.gz", binary=True)
    embedding_weights = torch.FloatTensor(w2v_model.vectors)
    embeddings = torch.nn.Embedding.from_pretrained(embedding_weights)
    embeddings.to(device)

    print("Initailizing model...")
    # initialize the model
    model = RNNModel(embedding_weights=embedding_weights,
                     hidden_size=hidden_dim,
                     num_layers=num_layers).to(device)

    print("Initializing optimizer and loss function...")
    # initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # initialize the loss function (BCE with logits loss)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # initialize the summary writer
    writer = SummaryWriter()

    train_accuracies: Dict[int, float]
    val_accuracies: Dict[int, float]

    print("STARTING TRAINING PROCESS...")
    # train the model
    (
        train_accuracies,
        val_accuracies
    ) = train_torch_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=loss_fn,
        optimizer=optimizer,
        epochs=epochs,
        print_every=print_every,
        patience=patience,
        writer=writer,
        embedding_weights=embedding_weights,
        device=device)

    print("Saving model...")
    # save the model
    save_model(model, f"{dataset_name}_DE_best_model")
    save_model(model, model_name)

    print("Plotting accuracies...")
    # plot the accuracies
    plot_accuracies(model,
                    train_dataloader,
                    val_dataloader,
                    test_dataloader,
                    embeddings,
                    train_accuracies,
                    val_accuracies,
                    model_name,
                    device=device)

    return None


if __name__ == "__main__":
    main()
