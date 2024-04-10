# deep learning libraries
import torch


class RNNModel(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        """
        This method is the constructor of the class.

        Args:
            hidden_size: hidden size of the RNN layers
        """

        # TODO
        super().__init__()
        torch.set_default_dtype(torch.double)  # set dtype to float64

        self.lstm = torch.nn.RNN(24, hidden_size, batch_first=True).cuda()
        self.fc = torch.nn.Linear(hidden_size, 24).cuda()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the model.

        Args:
            inputs: inputs tensor. Dimensions: [batch, number of past days, 24].

        Returns:
            output tensor. Dimensions: [batch, 24].
        """

        batch_size = inputs.size(0)
        # sequence = inputs.size(1)

        hidden_size = self.lstm.hidden_size

        """ IMOPRTANT TO SEND INITIAL STATES TO DEVICE """
        h0 = torch.zeros(1, batch_size, hidden_size,
                         dtype=torch.double).cuda()  # dtype=torch.float64

        lstm_output, _ = self.rnn(inputs, (h0,))

        output = self.fc(lstm_output[:, -1, :])

        return output
