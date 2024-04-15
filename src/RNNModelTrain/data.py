import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from gensim.models.keyedvectors import load_word2vec_format

import re

import string

import pandas as pd


from typing import List, Tuple


def tokenize_sentence(input_string: str) -> List[str]:
    def remove_punctuations(input_col):
        """To remove all the punctuations present in the text.Input the text column"""
        table = str.maketrans("", "", string.punctuation)
        return input_col.translate(table)

    input_string = remove_punctuations(input_string)
    input_string = re.sub(r"[^A-Za-z0-9(),.!?\'`\-\"]", " ", input_string)
    input_string = re.sub(r"\'s", " 's", input_string)
    input_string = re.sub(r"\'ve", " 've", input_string)
    input_string = re.sub(r"n\'t", " n't", input_string)
    input_string = re.sub(r"\'re", " 're", input_string)
    input_string = re.sub(r"\'d", " 'd", input_string)
    input_string = re.sub(r"\'ll", " 'll", input_string)
    input_string = re.sub(r"\.", " . ", input_string)
    input_string = re.sub(r",", " , ", input_string)
    input_string = re.sub(r"!", " ! ", input_string)
    input_string = re.sub(r"\?", " ? ", input_string)
    input_string = re.sub(r"\(", " ( ", input_string)
    input_string = re.sub(r"\)", " ) ", input_string)
    input_string = re.sub(r"\-", " - ", input_string)
    input_string = re.sub(r"\"", ' " ', input_string)
    # We may have introduced double spaces, so collapse these down
    input_string = re.sub(r"\s{2,}", " ", input_string)
    return list(filter(lambda x: len(x) > 0, input_string.split(" ")))


def tokenize_tweet(tweet: str) -> List[str]:
    """
    Tokenizes a given tweet by splitting the text into words, and doing any cleaning, replacing or normalization deemed useful

    Args:
        tweet (str): The tweet text to be tokenized.

    Returns:
        List[str]: A list of strings, representing the tokenized components of the tweet.
    """
    USR_MENTION_TOKEN = "<!USR_MENTION>"
    URL_TOKEN = "<!URL>"
    tweet = re.sub(r'@\w+', USR_MENTION_TOKEN, tweet)

    # replace the urls with a specific token
    tweet = re.sub(r'http\S+', URL_TOKEN, tweet)

    # remove the hashtags from the tweet
    tweet = re.sub(r'#\w+', '', tweet)

    return tweet.split()


def generate_review_text_target_pairs(file_path: str) -> Tuple[List[List[str]], List[int]]:
    """
    Generate a list of tokenized reviews and a list of target labels from the given file.

    Args:
        file_path (str): The path to the file.

    Returns:
        Tuple[ List[List[str]], List[int] ]: A tuple containing the list of tokenized reviews and the list of target labels.
    """
    texts: List[List[str]] = []
    targets: List[int] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            split_line = line.strip().split("\t")

            targets.append(int(split_line[-1]))

            # Join the words back together
            sentence = " ".join(split_line[:-1])
            sentence = tokenize_sentence(sentence)

            texts.append(sentence)

    return texts, targets


def generate_tweet_text_target_pairs(file_path: str) -> Tuple[List[List[str]], List[int]]:
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


class TextClassificationDataset(Dataset):
    """
    A PyTorch Dataset class for text classification tasks.

    Attributes:
        texts (List[List[str]]): List of the tokenized texts.
        targets (List[str]): List of the target labels.
    """

    def __init__(self, texts: List[List[str]], targets: List[str]) -> None:
        """
        Initializes the TextClassificationDataset with the given texts and targets.

        Args:
            texts (List[List[str]]): List of tokenized texts.
            targets (List[str]): List of target labels.
        """
        super().__init__()
        self.texts = texts
        self.targets = targets
        self._len = len(self.texts)

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return self._len

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the item at the given index.

        Args:
            index (int): Index of the item.

        Returns:
            Tuple[List[str], str]: Tuple of the text and the target label.
        """
        return self.texts[index], self.targets[index]


def generate_datasets(
    save_path: str = "./NLP_Data/data",
    dataset_name: str = "IMDB"
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Generate the training, validation, and test datasets for the specified dataset.

    Args:
        dataset_name (str, optional): Database name. Defaults to "IMDB".
            * "IMDB": IMDB movie review dataset.
            * "TweepFake": TweepFake dataset.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: train, validation and test datasets.
    """
    if not dataset_name in ["IMDB", "TweepFake"]:
        raise ValueError(
            "Invalid dataset name. Please choose from 'IMDB' or 'TweepFake'.")

    if dataset_name == "IMDB":
        train_val_texts, train_val_targets = generate_review_text_target_pairs(
            save_path + "/train.txt")

        split_point = int(len(train_val_texts) * 0.8)

        # Split the training set into training and validation sets
        train_texts, val_texts = train_val_texts[:
                                                 split_point], train_val_texts[split_point:]
        train_targets, val_targets = train_val_targets[:
                                                       split_point], train_val_targets[split_point:]

        test_texts, test_targets = generate_review_text_target_pairs(
            save_path + "/test.txt")

    if dataset_name == "TweepFake":
        train_texts, train_targets = generate_tweet_text_target_pairs(
            save_path + "/train.csv")
        val_texts, val_targets = generate_tweet_text_target_pairs(
            save_path + "/validation.csv")
        test_texts, test_targets = generate_tweet_text_target_pairs(
            save_path + "/test.csv")

    # create the datasets
    train_dataset = TextClassificationDataset(train_texts, train_targets)
    val_dataset = TextClassificationDataset(val_texts, val_targets)
    test_dataset = TextClassificationDataset(test_texts, test_targets)

    return train_dataset, val_dataset, test_dataset


def load_data(
    save_path: str = "./NLP_Data/data",
    dataset_name: str = "IMDB",
    batch_size: int = 64
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load the IMDBMovieReview dataset from the specified path, and split it into training, validation, and test sets.

    Args:
        save_path (str): The path to save the dataset.


    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: DataLoaders for the training, validation, and test sets.
    """

    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    (
        train_dataset,
        val_dataset,
        test_dataset
    ) = generate_datasets(save_path=save_path, dataset_name=dataset_name)

    # load the embeddings
    global w2v_model
    w2v_model = load_word2vec_format(
        "./NLP_Data/embeddings/GoogleNews-vectors-negative300.bin.gz", binary=True)

    # generate the dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


def word2idx(embedding_model, review: List[str]) -> torch.Tensor:
    """ 
    Converts a movie review to a list of word indices based on an embedding model.

    This function iterates through each word in the review and retrieves its corresponding index
    from the embedding model's vocabulary. If a word is not present in the model's vocabulary,
    it is skipped.

    Args:
        embedding_model (Any): The embedding model with a 'key_to_index' attribute, which maps words to their indices.
        review (List[str]): A list of words representing the tweet.

    Returns:
        torch.Tensor: A tensor of word indices corresponding to the words in the tweet.
    """
    indices = [embedding_model.key_to_index[word] if word in embedding_model.key_to_index else 0
           for word in review]

    return torch.tensor(indices)


def collate_fn(batch: List[Tuple[List[str], int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for the DataLoader.

    Args:
        batch (List[Tuple[List[str], int]]): A list of tuples containing the reviews and target labels.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the reviews and target labels.
            - texts_padded (torch.Tensor): A tensor of padded word indices of the text.
            - labels (torch.Tensor): A tensor of target labels.
            - lengths (torch.Tensor): A tensor of lengths of each review.
    """

    global w2v_model

    # Sort the batch by the length of the reviews
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

    # Separate the reviews and target labels
    texts: List[List[str]]
    labels: List[int]
    texts, labels = zip(*batch)

    # Convert the reviews to bag of words representation
    texts_idx: List[torch.Tensor] = [
        word2idx(w2v_model, review) for review in texts]

    lengths: List[torch.Tensor] = [max(len(text), 1) for text in texts_idx]

    # Convert the lengths to a tensor
    lengths: torch.Tensor = torch.tensor(lengths, dtype=torch.float32)

    # Pad the reviews
    texts_padded: torch.Tensor = pad_sequence(
        texts_idx, batch_first=True)

    # Convert the target labels to a tensor
    labels: torch.Tensor = torch.tensor(labels, dtype=torch.float32)

    return texts_padded, labels, lengths
