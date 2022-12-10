import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer


def get_biased_model(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_state = torch.load('base_model_params.pt', map_location=device)['model_state_dict']
    model = HypOnly(4096, 4096, 3).to(device)
    model.load_state_dict(model_state)
    return model

class BERTNLIModel(nn.Module):
    def __init__(self,
                 bert_model,
                 output_dim,
                 ):
        super().__init__()

        self.bert = bert_model
        embedding_dim = bert_model.config.to_dict()['hidden_size']
        self.out = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, output_dim))

    def forward(self, sequence, attn_mask, token_type):
        embedded = self.bert(input_ids=sequence, attention_mask=attn_mask, token_type_ids=token_type)[1]
        # print(embedded.shape)
        output = self.out(embedded)
        # print(output.shape)
        return output


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


# Uses BERT Transformer with a biased model as the loss
class UnBiasedModel(nn.Module):
    def __init__(self, biased_model, device=None):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param d_embed: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param d_output: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        model_name = 'bert-base-uncased'
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(
            device)
        # self.IE = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_embed)
        self.biased_model = biased_model

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else None
    
        _, pooledOut = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        _, pooledOut_biased = self.biased_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        print(pooledOut,pooledOut_biased)
        bertOut = self.bert_drop(pooledOut)
        output = self.out(bertOut)

        return output
