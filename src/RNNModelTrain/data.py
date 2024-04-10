# import necessary dependencies
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from gensim.models.keyedvectors import load_word2vec_format

import pandas as pd

import re

from typing import List, Tuple, Any


def tokenize_tweet(tweet: str) -> List[str]:
    """
      Tokenizes a given tweet by splitting the text into words, and doing any cleaning, replacing or normalization deemed useful

      Args:
          tweet (str): The tweet text to be tokenized.

      Returns:
          list[str]: A list of strings, representing the tokenized components of the tweet.
    """
    # TODO: Complete the tokenize_tweet function
    # replace the usernas with a specific token
    global USR_MENTION_TOKEN
    global URL_TOKEN
    USR_MENTION_TOKEN = "<!USR_MENTION>"
    URL_TOKEN = "<!URL>"
    tweet = re.sub(r'@\w+', USR_MENTION_TOKEN, tweet)

    # replace the urls with a specific token
    tweet = re.sub(r'http\S+', URL_TOKEN, tweet)

    # remove the hashtags from the tweet
    tweet = re.sub(r'#\w+', '', tweet)

    return tweet.split()


def generate_text_target_pairs(file_path: str) -> Tuple[List[List[str]], List[int]]:
    """
    Load data from a specified file path, extract texts and targets, and tokenize the texts using the tokenize_tweet function.

    Parameters:
    file_path (str): The path to the dataset file.

    Returns:
    Tuple[List[str], List[int]]: Lists of texts and corresponding targets.
    """
    try:
        # TODO: Read the corresponding csv
        data: pd.DataFrame = pd.read_csv(file_path)

        # TODO: Obtain the text column from data
        texts: List[str] = data['text'].tolist()

        # TODO: Obtain targets, 0 for human and 1 for bot
        # replace the target column with a binary representation
        data['tag'] = data['account.type'].replace('human', 0)
        data['tag'] = data['tag'].replace('bot', 1)
        targets: List[int] = data['tag'].tolist()

        # TODO: Return tokenized texts, and targets
        return [tokenize_tweet(text) for text in texts], targets

    except FileNotFoundError:
        print(f"{file_path} not found. Please check the file path.")


class TweepFakeDataset(Dataset):
    """
    A PyTorch Dataset for the TweepFake dataset.

    Attributes:
        texts (List[List[str]]): List of tweets tokens.
        targets (List[str]): List of target labels.
    """

    def __init__(self,
                 texts: List[List[str]],
                 targets: List[int]
                 ):
        """
        Initializes the TweepFakeDataset with the given file path.

        Args:
            texts (List[List[str]]): List of tweets tokens.
            targets (List[str]): List of target labels.
        """
        # TODO: Complete the init function
        # initialise the super class
        super().__init__()
        self.texts = texts
        self.targets = targets
        self._len = len(self.texts)

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        # TODO: Complete the len function
        return self._len  # VALID IF WE DONT UPDATE THE DATASET, OTHERWISE WOULD NEED TO IMPLEMENT AN _update_len METHOD

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the embedded tensor and target for the text at the specified index.

        Args:
            idx (int): Index of the item.

        Returns:
            Tuple[List[str], List[int]]: A tuple containing the BoW vector and the target label.
        """
        # TODO: Complete the getitem function

        return self.texts[idx], self.targets[idx]


def load_data(
    save_path: str = "./NLP_Data/data",
    batch_size: int = 64
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load the TweepFake dataset from the specified path, and split it into training, validation, and test sets.

    Args:
        save_path (str): The path to save the dataset.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: DataLoaders for the training, validation, and test sets.
    """

    # get the initial data
    train_texts, train_targets = generate_text_target_pairs(
        save_path + "/train.csv")
    val_texts, val_targets = generate_text_target_pairs(
        save_path + "/validation.csv")
    ts_texts, ts_targets = generate_text_target_pairs(save_path + "/test.csv")

    # create the datasets
    train_dataset = TweepFakeDataset(train_texts, train_targets)
    val_dataset = TweepFakeDataset(val_texts, val_targets)
    test_dataset = TweepFakeDataset(ts_texts, ts_targets)

    #! CREATE THE GLOBAL W2V MODEL
    global w2v_model  # ! THIS IS NOT THE BEST PRACTICE, BUT IT IS USED HERE FOR SIMPLICITY
    w2v_model = load_word2vec_format(
        "./NLP_Data/embeddings/GoogleNews-vectors-negative300.bin.gz", binary=True)

    # create the dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, test_dataloader


def collate_fn(batch: List[Tuple[List[str], int]],
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepares and returns a batch for training/testing in a torch model.

    This function sorts the batch by the length of the text sequences in descending order,
    tokenizes the text using a pre-defined word-to-index mapping, pads the sequences to have
    uniform length, and converts labels to tensor.

    Args:
        batch (List[Tuple[List[str], int]]): A list of tuples, where each tuple contains a
                                             list of words (representing a text) and an integer label.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing three elements:
            - texts_padded (torch.Tensor): A tensor of padded word indices of the text.
            - labels (torch.Tensor): A tensor of labels.
            - lengths (torch.Tensor): A tensor representing the lengths of each text sequence.
    """
    # get the w2v model from the global scope
    global w2v_model

    # TODO: Sort the batch by the length of text sequences in descending order
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

    # TODO: Unzip texts and labels from the sorted batch
    texts: List[str]
    labels: List[int]
    texts, labels = zip(*batch)

    # convert the elements of the labels list to int
    labels = list(labels)
    labels = [int(label) for label in labels]

    # TODO: Convert texts to indices using the word2idx function and w2v_model
    texts_indx: List[torch.Tensor] = [
        word2idx(w2v_model, tweet) for tweet in texts]

    # TODO: Calculate the lengths of each element of texts_indx.
    # The minimum length shall be 1, in order to avoid later problems when training the RNN
    lengths: List[torch.Tensor] = [max(len(tweet), 1) for tweet in texts_indx]

    # TODO: Pad the text sequences to have uniform length
    texts_padded: torch.Tensor = pad_sequence(texts_indx, batch_first=True)

    # TODO: Convert labels to tensor
    labels: torch.Tensor = torch.tensor(labels)

    return texts_padded, labels, lengths


def word2idx(embedding_model: Any, tweet: List[str]) -> torch.Tensor:
    """
    Converts a tweet to a list of word indices based on an embedding model.

    This function iterates through each word in the tweet and retrieves its corresponding index
    from the embedding model's vocabulary. If a word is not present in the model's vocabulary,
    it is skipped.

    Args:
        embedding_model (Any): The embedding model with a 'key_to_index' attribute, which maps words to their indices.
        tweet (List[str]): A list of words representing the tweet.

    Returns:
        torch.Tensor: A tensor of word indices corresponding to the words in the tweet.
    """
    # TODO: Complete the function according to the requirements

    # get the indices of the words in the tweet
    indices = [embedding_model.key_to_index[word]
               for word in tweet if word in embedding_model.key_to_index]

    return torch.tensor(indices)
