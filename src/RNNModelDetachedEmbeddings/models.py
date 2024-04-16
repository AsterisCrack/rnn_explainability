import torch


class RNNModel(torch.nn.Module):
    def __init__(self,
                 embedding_weights: torch.Tensor,
                 hidden_size: int,
                 num_layers: int
                 ) -> None:
        """
        This method is the constructor of the class.

        Args:
            embedding_weights: weights for the embedding layer
            hidden_size: size of the hidden state of the RNN
            num_layers: number of RNN layers
            device: device to run the model
        """

        # TODO
        super().__init__()
        torch.set_default_dtype(torch.float32)  # set dtype to float32

        embedding_dim = embedding_weights.shape[1]

        # self.embedding: torch.nn.Embedding = torch.nn.Embedding.from_pretrained(embedding_weights)

        self.rnn: torch.nn.RNN = torch.nn.RNN(
            embedding_dim, hidden_size, num_layers, batch_first=True)

        self.fc: torch.nn.Linear = torch.nn.Linear(hidden_size, 1)

    def forward(self, embedded_inputs: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the model.

        Args:
            embedded_inputs: embedding of the inputs tensor.
            text_lengths: lengths of the texts.

        Returns:
            output tensor. Dimensions: [batch, ].
        """

        # embedded: torch.Tensor = self.embedding(inputs)

        packed_embedding: torch.nn.utils.rnn.PackedSequence = torch.nn.utils.rnn.pack_padded_sequence(
            embedded_inputs, text_lengths, batch_first=True, enforce_sorted=False)

        packed_output, hidden = self.rnn(packed_embedding)

        last_hidden: torch.Tensor = hidden[-1]

        return self.fc(last_hidden).squeeze(1)
