import torch
import torch.nn as nn

class HypOnly(nn.Module):
    # Just a FF Network
    def __init__(self, d_embed, d_hidden, d_output):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param d_embed: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param d_output: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        # self.IE = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_embed)
        self.prediction = nn.Sequential(
            nn.Linear(d_embed, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_output),
            nn.LogSoftmax(dim=2)
        )
        self.prediction = nn.Linear(d_embed, d_output)  # only using one layer

    def forward(self, input_: torch.tensor):
        """

        :param input_: a sequence of size (num_examples, seq_len , 1) or (seq_len,)
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        if input_.is_cuda:
            device = input_.get_device()
        else:
            device = torch.device("cpu")
        # print(device)
        output = self.prediction(input_).to(device)

        return output