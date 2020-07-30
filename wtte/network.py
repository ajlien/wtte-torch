# Classes for generating a WTTE-RNN network for training and inference

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np

class WeibullActivation(nn.Module):
    """Activation function to get elementwise alpha and regularized beta
    :param init_alpha: Float initial alpha value
    :param max_beta_value: Float maximum beta value
    """
    def __init__(self, init_alpha=1.0, max_beta_value=5.0, scalefactor=1.0):
        super(WeibullActivation, self).__init__()
        self.init_alpha = init_alpha
        self.max_beta_value = max_beta_value
        self.scalefactor = scalefactor

    def forward(self, x):
        a, b = torch.split(x, 1, dim=-1)
        a = self.init_alpha * torch.exp(a) * self.scalefactor
        b = self.max_beta_value * torch.sigmoid(b) * self.scalefactor
        return torch.cat((a, b), -1)


class StubModel(nn.Module):
    """Make a stub model for pretraining"""
    def __init__(self, linear, activation):
        super(StubModel, self).__init__()
        self.linear = linear
        self.activation = activation
    
    def forward(self, x):
        y, _ = pad_packed_sequence(x, batch_first=True, padding_value=-99)
        y = y.new_zeros((y.shape[0], y.shape[1], self.linear.in_features))
        y = self.linear(y)
        y = self.activation(y)
        return y[:,-1,:]


class WtteNetwork(nn.Module):
    """A deep neural network that receives a sequence as input and estimates the parameters of a
    Weibull distribution, conditional on inputs, describing the time to next event of the process.
    """
    def __init__(self, submodel, submodel_out_features, init_alpha=1.0, max_beta_value=5.0):
        """
        :param submodel: nn.Module with the architecture of the model
        :param submodel_out_features: Int with the dimension of the output from the last layer of the submodel
        :param init_alpha: Float initial alpha value
        :param max_beta_value: Float maximum beta value
        """
        super(WtteNetwork, self).__init__()
        self.init_alpha = init_alpha
        self.max_beta_value = max_beta_value
        self.submodel = submodel
        self.linear = nn.Linear(submodel_out_features, 2)
        self.activation = WeibullActivation(init_alpha=init_alpha, max_beta_value=max_beta_value, 
                                            scalefactor=1.0/np.log(submodel_out_features))

    def forward(self, x):
        y = self.submodel(x)
        y = self.linear(y)
        y = self.activation(y)
        return y[:,-1,:]


class WtteRnnNetwork(WtteNetwork):
    """A network with recurrent layers that estimates the Weibull time to event distribution parameters.
    Default architecture based on Egil Martinsson's example at
    https://github.com/ragulpr/wtte-rnn/blob/master/examples/keras/simple_example.ipynb
    """
    def __init__(self, input_size, rnn_layer=nn.GRU, 
                 rnn_layer_options={'hidden_size': 20, 'num_layers': 1},
                 init_alpha=1.0, max_beta_value=5.0):
        """Specify an RNN for WTTE modeling.
        :param input_size: Int sequence length provided
        :param rnn_layer: Class with the nn.Module representing the recurrent layer - consider nn.GRU or nn.LSTM
        :param rnn_layer_options: Dict with parameters for RNN hidden layer
        :param init_alpha: Float initial alpha value
        :param max_beta_value: Float maximum beta value
        """
        rnn = rnn_layer(input_size=input_size, batch_first=True, **rnn_layer_options)
        super(WtteRnnNetwork, self).__init__(submodel=rnn, submodel_out_features=rnn.hidden_size, 
                                             init_alpha=init_alpha, max_beta_value=max_beta_value)

    def forward(self, x):
        y, _ = self.submodel(x)
        y, _ = pad_packed_sequence(y, batch_first=True, padding_value=0)
        y = self.linear(y)
        y = self.activation(y)
        return y[:,-1,:]
