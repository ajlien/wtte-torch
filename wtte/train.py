# Helper functions for training models

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
import logging
from wtte.loss import loss_continuous_weibull_loglik, loss_discrete_weibull_loglik
from wtte.network import StubModel
import numpy as np

"""
https://www.analyticsvidhya.com/blog/2019/01/guide-pytorch-neural-networks-case-studies/
"""

def train(model, train_dataloader, test_dataloader=None, n_epochs=500, lr=0.01, clip_grad=None, 
          loss_type='discrete', pretrain=True, device=torch.device('cpu')):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if loss_type == 'discrete':
        wtte_loss = loss_discrete_weibull_loglik
    elif loss_type == 'continuous':
        wtte_loss = loss_continuous_weibull_loglik
    else:
        raise ValueError('loss_type must be "discrete" or "continuous"')
    if pretrain:
        logging.info('Begin pretraining')
        temp_model = StubModel(model.linear, model.activation)
        xinit, yinit = iter(train_dataloader).next()
        xinit = xinit.to(device)
        init_bias = temp_model(xinit)[0,:].view(2).detach().cpu().numpy()
        logging.info('Initial values: alpha {:0.3f}, beta {:0.3f}'.format(init_bias[0], init_bias[1]))
        for epoch in range(n_epochs // 4):
            pretrain_loss = []
            with torch.set_grad_enabled(True):
                for x, yu in train_dataloader:
                    x, yu = x.to(device), yu.to(device)
                    optimizer.zero_grad()
                    ab = temp_model(x)
                    loss = wtte_loss(yu, ab)
                    loss.backward()
                    if clip_grad is not None:
                        clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                    optimizer.step()
                    pretrain_loss.append(loss.item())
            msg_out = 'Epoch {} of {}: Pretraining Loss {}'.format(epoch+1, n_epochs, np.mean(pretrain_loss))
            logging.info(msg_out)
        init_bias = temp_model(xinit)[0,:].view(2).detach().cpu().numpy()
        logging.info('Pretrained values: alpha {:0.3f}, beta {:0.3f}'.format(init_bias[0], init_bias[1]))
    logging.info('Begin training')
    for epoch in range(n_epochs):
        train_loss = []
        with torch.set_grad_enabled(True):
            for x, yu in train_dataloader:
                x, yu = x.to(device), yu.to(device)
                optimizer.zero_grad()
                ab = model(x)
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
                for x, yu in train_dataloader:
                    x, yu = x.to(device), yu.to(device)
                    ab = model(x)
                    loss = loss_continuous_weibull_loglik(yu, ab)
                    test_loss.append(loss.item())
            msg_out += ', Test Loss {}'.format(np.mean(test_loss))
        logging.info(msg_out)
    
