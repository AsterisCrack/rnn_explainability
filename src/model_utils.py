# import necessary dependencies
import torch
from torch.jit import RecursiveScriptModule
from torch.utils.data import DataLoader
# Import sigmoid function
from torch.nn.functional import sigmoid
from torch.nn.utils.rnn import pad_sequence
from src.RNNModelTrain.data import tokenize_tweet
from src.RNNModelTrain.data import tokenize_sentence
from typing import List, Tuple, Any, Union

from gensim.models.keyedvectors import load_word2vec_format

import numpy as np
import random
import os
import matplotlib.pyplot as plt

global w2v_model
w2v_model = None


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


def load_w2v_model() -> Any:
    """
    This function loads a pre-trained word2vec model from a given file path.

    Args:
        file_path: path to the pre-trained word2vec model.
    """

    #! CREATE THE GLOBAL W2V MODEL
    global w2v_model  # ! THIS IS NOT THE BEST PRACTICE, BUT IT IS USED HERE FOR SIMPLICITY
    if w2v_model is None:
        w2v_model = load_word2vec_format(
            "./NLP_Data/embeddings/GoogleNews-vectors-negative300.bin.gz", binary=True)
    return w2v_model


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

def predict_single_text(
        text: str, model: torch.nn.Module, device: str = 'cpu', probability: bool = False, model_type: str = "IMDB", likelihood=False) -> Union[float, int]:
    
    if type(device) == str:
        device = torch.device(device)
    model.to(device)
    model.eval()
    if model_type == "IMDB":
        tokenized_text = tokenize_sentence(text)
    else:
        tokenized_text = tokenize_tweet(text)

    # Collate the text
    w2v_model = load_w2v_model()
    tokenized_text = word2idx(w2v_model, tokenized_text)
    text_padded = pad_sequence([tokenized_text], batch_first=True)
    length = torch.tensor([len(tokenized_text)])
    if length == 0:
        return 0
    # Send to device
    text_padded = text_padded.to(device)
    prediction = model(text_padded, length)

    if probability:
        return sigmoid(prediction).item()
    if likelihood:
        return prediction.item()
    prediction = (prediction > 0.5).item()
    return 1 if prediction else 0


def predict_multiple_text(
    texts: List[str], model: torch.nn.Module, device: str = 'cpu', probability: bool = False, model_type: str = "IMDB", likelihood=False
) -> List[int]:
    
    if type(device) == str:
        device = torch.device(device)
    predictions = [predict_single_text(text, model, device, probability=probability,
                                       model_type=model_type, likelihood=likelihood) for text in texts]

    return predictions


def predict_single_text_DE(text: str,
                           model: torch.nn.Module,
                           embedding: torch.nn.Embedding,
                           device: str = 'cpu',
                           probability: bool = False,
                           model_type: str = "IMDB",
                           likelihood=False) -> int:
    """
    A function to predict the sentiment of a single text using a model without an 
    internal embedding layer. 

    Args:
        text (str): The text to predict the sentiment for.
        model (torch.nn.Module): The model used for prediction.
        embedding (torch.nn.Embedding): The embedding layer to be applied to the input texts.
        device (str): The device to run the model on.
        probability (bool): Whether to return the probability of the prediction.
        model_type (str): The type of model used for prediction.
        likelihood (bool): Whether to return the likelihood of the prediction.

    Returns:
        int: The predicted sentiment of the text.
    """
    
    if type(device) == str:
        device = torch.device(device)
        
    model.to(device)
    model.eval()
    if model_type == "IMDB":
        tokenized_text = tokenize_sentence(text)
    else:
        tokenized_text = tokenize_tweet(text)

    # Collate the text
    w2v_model = load_w2v_model()
    tokenized_text = word2idx(w2v_model, tokenized_text)
    text_padded = pad_sequence([tokenized_text], batch_first=True)
    length = torch.tensor([len(tokenized_text)])
    if length == 0:
        return 0
    # Send to device
    text_padded = text_padded.to(device)
    embedding.to(device)

    # embed the text
    embedded_text = embedding(text_padded)

    prediction = model(embedded_text, length)

    if probability:
        return sigmoid(prediction).item()
    if likelihood:
        return prediction.item()
    prediction = (prediction > 0.5).item()
    return 1 if prediction else 0


def predict_multiple_text_DE(texts: List[str],
                             model: torch.nn.Module,
                             embedding: torch.nn.Embedding,
                             device: str = 'cpu',
                             probability: bool = False,
                             model_type: str = "IMDB",
                             likelihood=False) -> List[int]:
    """
    A function to predict the sentiment of multiple texts using a model without an
    internal embedding layer.

    Args:
        texts (List[str]): The list of texts to predict the sentiment for.
        model (torch.nn.Module): The model used for prediction.
        embedding (torch.nn.Embedding): The embedding layer to be applied to the input texts.
        device (str, optional): Device to be used for the task. Defaults to 'cpu'.
        probability (bool, optional): If True, returns the probability of the prediction. Defaults to False.
        model_type (str, optional): Database that the model will predicting on. Defaults to "IMDB".
        likelihood (bool, optional): . Defaults to False.

    Returns:
        List[int]: _description_
    """

    if type(device) == str:
        device = torch.device(device)
        
    predictions = [predict_single_text_DE(text, model, embedding, device, probability=probability,
                                       model_type=model_type, likelihood=likelihood) for text in texts]

    return predictions
