import torch


class RNNModel(torch.nn.Module):
    def __init__(self, embedding_weights: torch.Tensor, hidden_size: int, num_layers: int) -> None:
        """
        This method is the constructor of the class.

        Args:
            embedding_weights: weights for the embedding layer
            hidden_size: hidden size of the RNN layers
            num_layers: number of RNN layers
            device: device to run the model
        """

        # TODO
        super().__init__()
        torch.set_default_dtype(torch.float32)  # set dtype to float32

        embedding_dim = embedding_weights.shape[1]

        self.embedding = torch.nn.Embedding.from_pretrained(
            embedding_weights)

        self.rnn = torch.nn.RNN(
            embedding_dim, hidden_size, num_layers, batch_first=True)

        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the model.

        Args:
            inputs: inputs tensor. Dimensions: [batch, number of past days, 24].

        Returns:
            output tensor. Dimensions: [batch, 24].
        """

        embedded: torch.Tensor = self.embedding(inputs)

        packed_embedded: torch.nn.utils.rnn.PackedSequence = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths, batch_first=True, enforce_sorted=False)

        packed_output, hidden = self.rnn(packed_embedded)

        hidden: torch.Tensor = hidden[-1]

        return self.fc(hidden).squeeze(1)
