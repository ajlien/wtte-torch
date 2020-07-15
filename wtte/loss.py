# Specialized loss functions for censored Weibull-distributed time to event data

import torch
from math import log


def loss_continuous_weibull_loglik(yu, ab, clip_prob=1e-5, eps=1e-16):
    """Continuous Weibull log-likelihood loss
    :param yu: Tensor with last dimension of size 2, corresponding to TTE and to censoring indicator
    :param ab: Tensor with last dimension of size 2, corresponding to estimated alpha and beta parameters
    :param clip_prob: Float for clipping likelihood to [log(clip_prob), log(1-clip_prob)]
    :param eps: Float tiny epsilon for stability
    """
    y, u = torch.split(yu, 1, dim=-1)
    a, b = torch.split(ab, 1, dim=-1)

    ya = (y + eps) / a
    loglik = u * (torch.log(b) + b * torch.log(ya)) - torch.pow(ya, b)
    if clip_prob is not None:
        loglik = torch.clamp(loglik, log(clip_prob), log(1 - clip_prob))
    return -1 * torch.mean(loglik)


def loss_discrete_weibull_loglik(yu, ab, clip_prob=1e-5, eps=1e-16):
    """Discrete Weibull log-likelihood loss
    :param yu: Tensor with last dimension of size 2, corresponding to TTE and to censoring indicator
    :param ab: Tensor with last dimension of size 2, corresponding to estimated alpha and beta parameters
    :param clip_prob: Float for clipping likelihood to [log(clip_prob), log(1-clip_prob)]
    :param eps: Float tiny epsilon for stability
    """
    y, u = torch.split(yu, 1, dim=-1)
    a, b = torch.split(ab, 1, dim=-1)

    hazard0 = torch.pow((y + eps) / a, b)
    hazard1 = torch.pow((y + 1.0) / a, b)

    loglik = u * torch.log(torch.exp(hazard1 - hazard0) - (1.0 - eps)) - hazard1
    if clip_prob is not None:
        loglik = torch.clamp(loglik, log(clip_prob), log(1 - clip_prob))
    return -1 * torch.mean(loglik)