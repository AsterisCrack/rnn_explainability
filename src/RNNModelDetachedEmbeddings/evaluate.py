import torch

from src.RNNModelDetachedEmbeddings.train_functions import t_step
from src.RNNModelDetachedEmbeddings.utils import load_model, set_seed
from src.RNNModelDetachedEmbeddings.data import load_data

from typing import Final

# import the load_word2vec_format function
from gensim.models.keyedvectors import load_word2vec_format

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 2222
set_seed(SEED)

DATA_PATH: Final[str] = "./NLP_Data/data"


def main() -> None:
    # load model
    print("Loading model...")
    model = load_model(f"DE_best_model")

    # load the data
    print("Loading data...")
    (
        train_dataloader,
        val_dataloader,
        test_dataloader
    ) = load_data(
        save_path=DATA_PATH)

    # load embeddings
    print("Loading embeddings...")
    w2v_model = load_word2vec_format(
        "./NLP_Data/embeddings/GoogleNews-vectors-negative300.bin.gz", binary=True)
    embedding_weights = torch.FloatTensor(w2v_model.vectors)
    embeddings = torch.nn.Embedding.from_pretrained(embedding_weights)
    embeddings.to(DEVICE)

    # perform testing
    print("Testing model...")
    test_accuracy = t_step(model, test_dataloader, embeddings, device=DEVICE)

    print(f"Test accuracy: {test_accuracy:.4f}")

    return


if __name__ == "__main__":
    main()
