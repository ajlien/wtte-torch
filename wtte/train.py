# Helper functions for training models

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_packed_sequence
import logging
from wtte.loss import loss_continuous_weibull_loglik, loss_discrete_weibull_loglik
from wtte.network import StubModel
import numpy as np
from tqdm import tqdm

"""
https://www.analyticsvidhya.com/blog/2019/01/guide-pytorch-neural-networks-case-studies/
"""

def pretrain(model, train_dataloader, optimizer, wtte_loss, n_epochs=25, clip_grad=None, device=torch.device('cpu')):
    """Fit the biases for Weibull activation alpha and beta, 
       in practice this substantially helps model convergence
    """
    logging.info('Begin pretraining')
    temp_model = StubModel(model.linear, model.activation)
    xinit, yinit = iter(train_dataloader).next()
    xinit = xinit.to(device=device)
    init_bias = temp_model(xinit)[0,-1,:].view(2).detach().cpu().numpy()
    logging.info('Initial values: alpha {:0.3f}, beta {:0.3f}'.format(init_bias[0], init_bias[1]))
    for epoch in range(n_epochs):
        pretrain_loss = []
        with torch.set_grad_enabled(True):
            for x, yu in tqdm(train_dataloader, ascii=True):
                x, yu = x.to(device=device), yu.to(device=device)
                optimizer.zero_grad()
                ab = temp_model(x)
                yu, _ = pad_packed_sequence(yu, batch_first=True, padding_value=0)
                loss = wtte_loss(yu, ab)
                loss.backward()
                if clip_grad is not None:
                    clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                optimizer.step()
                pretrain_loss.append(loss.item())
        msg_out = 'Epoch {} of {}: Pretraining Loss {}'.format(epoch+1, n_epochs, np.mean(pretrain_loss))
        logging.info(msg_out)
    init_bias = temp_model(xinit)[0,-1,:].view(2).detach().cpu().numpy()
    logging.info('Pretrained values: alpha {:0.3f}, beta {:0.3f}'.format(init_bias[0], init_bias[1]))


def train(model, train_dataloader, test_dataloader=None, n_epochs=500, lr=0.01, clip_grad=None, 
          loss_type='discrete', n_epochs_pretrain=10, device=torch.device('cpu')):
    """Train a WTTE-RNN model such that error is evaluated at every (non-padded) time step.
    """
    _ = model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if loss_type == 'discrete':
        wtte_loss = loss_discrete_weibull_loglik
    elif loss_type == 'continuous':
        wtte_loss = loss_continuous_weibull_loglik
    else:
        raise ValueError('loss_type must be "discrete" or "continuous"')
    if n_epochs_pretrain is not None and n_epochs_pretrain > 0:
        pretrain(model, train_dataloader, optimizer, wtte_loss, n_epochs=n_epochs_pretrain, clip_grad=clip_grad, device=device)
    logging.info('Begin training')
    for epoch in range(n_epochs):
        train_loss = []
        with torch.set_grad_enabled(True):
            for x, yu in tqdm(train_dataloader, ascii=True):
                x, yu = x.to(device=device), yu.to(device=device)
                optimizer.zero_grad()
                ab = model(x)
                yu, _ = pad_packed_sequence(yu, batch_first=True, padding_value=0)
                loss = wtte_loss(yu, ab)
                loss.backward()
                if clip_grad is not None:
                    clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                optimizer.step()
                train_loss.append(loss.item())
        msg_out = 'Epoch {} of {}: Train Loss {}'.format(epoch+1, n_epochs, np.mean(train_loss))

        if test_dataloader is not None:
            test_loss = []
            with torch.set_grad_enabled(False):
                for x, yu in tqdm(train_dataloader, ascii=True):
                    x, yu = x.to(device=device), yu.to(device=device)
                    ab = model(x)
                    yu, _ = pad_packed_sequence(yu, batch_first=True, padding_value=0)
                    loss = wtte_loss(yu, ab)
                    test_loss.append(loss.item())
            msg_out += ', Test Loss {}'.format(np.mean(test_loss))
        logging.info(msg_out)
    
