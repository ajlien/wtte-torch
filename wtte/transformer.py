# Classes for WTTE-Transformer models

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np
from wtte.network import WeibullActivation, WtteNetwork
import logging


class PositionalEncoding(nn.Module):
    """Adds a channel with information about the relative position of observations in the sequence
    Adapted from David Pollack,
    PyTorch tutorial 'Sequence-to-Sequence Modeling with nn.Transformer and TorchText'
    https://github.com/pytorch/tutorials/blob/master/beginner_source/transformer_tutorial.py
    Commit 809f27e on Oct 13, 2019
    """
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class WtteAttentionNetwork(WtteNetwork):
    """A network with a Transformer architecture that estimates the Weibull time to event distribution parameters.
    See Vaswani et al (2017), "Attention Is All You Need", https://arxiv.org/abs/1706.03762
    Experiment with whether to use positional encoding and/or an attention mask.
    Adapted from David Pollack,
    PyTorch tutorial 'Sequence-to-Sequence Modeling with nn.Transformer and TorchText'
    https://github.com/pytorch/tutorials/blob/master/beginner_source/transformer_tutorial.py
    Commit 809f27e on Oct 13, 2019
    """
    def __init__(self, input_size, num_layers=2, norm=None,
                 encoder_layer_options={'nhead': 8, 'dim_feedforward': 32, 'dropout': 0.1},
                 mask=True, positional_encoding=True,
                 init_alpha=1.0, max_beta_value=5.0):
        """Specify an RNN for WTTE modeling.
        :param input_size: Int sequence length provided
        :param num_layers: Int number of TransformerEncoderLayer layers to use
        :param norm: Optional layer normalization component
        :param encoder_layer_options: Dict with parameters for TransformerEncoderLayer
        :param mask: Boolean takes True if using a square mask to hide future observations from attention at each time step
        :param positional_encoding: Boolean takes True if using positional encoding features
        :param init_alpha: Float initial alpha value
        :param max_beta_value: Float maximum beta value
        """
        self.mask = mask
        self.src_mask = None
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, **encoder_layer_options)
        encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers, norm=norm)
        super(WtteAttentionNetwork, self).__init__(submodel=encoder, submodel_out_features=input_size, 
                                                   init_alpha=init_alpha, max_beta_value=max_beta_value)
        if positional_encoding:
            self.pos_encoder = PositionalEncoding(input_size, dropout=encoder_layer_options['dropout'])
        else:
            self.pos_encoder = nn.Identity()

    def generate_padding_mask(self, batch, lens):
        """Generate an encoder self-attention mask that ignores padding for a batch of sequences with varying lengths
        :param batch: Tensor of size (T, B, *) where T is the length of the longest sequence in batch
        :param lens: Tensor with list of lengths of each sequence in the batch
        Note that these params are the result of calling torch.nn.utils.rnn.pad_packed_sequence on a batch.
        """
        mask = torch.Tensor(np.arange(batch.shape[0])).to(device=batch.device)
        mask = mask.expand(batch.shape[1], -1)
        expand_lens = lens.to(device=batch.device).unsqueeze(1).expand(-1, batch.shape[0])
        mask = (mask >= expand_lens)  # True = ignore, False = attend
        return mask

    def generate_subsequent_mask(self, batch, lens):
        """Generate an encoder self-attention mask that ignores future observations in each sequence
        :param batch: Tensor of size (T, B, *) where T is the length of the longest sequence in batch
        :param lens: Tensor with list of lengths of each sequence in the batch
        Note that these params are the result of calling torch.nn.utils.rnn.pad_packed_sequence on a batch.
        """
        T = len(batch)
        mask = (torch.triu(batch.new_ones(T, T)) == 1).transpose(0, 1)
        #mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        # Unpack and pad the batch - TransformerEncoder can't use PackedSequence
        y, z = pad_packed_sequence(x, batch_first=False, padding_value=0)
        padmask = self.generate_padding_mask(y, z)
        futuremask = self.generate_subsequent_mask(y, z)
        y = self.pos_encoder(y)
        y = self.submodel(y, 
                          mask=futuremask,
                          src_key_padding_mask=padmask)
        y = y.transpose(1, 0)  # Put batch in first dimension again
        y = self.linear(y)
        y = self.activation(y)
        return y
