# Helper functions for performing inference with models

import torch
from torch.nn.utils.rnn import pad_packed_sequence
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

def unbatch_padded(x, lens_x):
    """Make a list of individual batch elements with padded (masked) entries omitted"""
    x_split = x.chunk(x.shape[0], dim=0)
    x_clean = [x_split[i].reshape(-1, 2)[:lens_x[i]].detach().cpu().numpy() 
               for i in range(len(lens_x))]
    return x_clean

def predict(model, dataloader, device=torch.device('cpu'), to_dataframe=False, final_only=False):
    """Generate predictions from a provided dataloader and model
    :param model: a WtteNetwork
    :param dataloader: a Pytorch DataLoader
    :param device: the device performing computations for Pytorch
    :param to_dataframe: Boolean takes True if output should be a pandas DataFrame,
        False if should be a list of tuples of tensors
    :param final_only: Boolean takes True if keep only the last time step from each example,
        False if entire sequences should be kept
    :return: if `to_dataframe`, a pandas DataFrame with combined results, otherwise
        a list of tuples - one per sequence in the dataloader - with elements
        (predictions, actual, identifiers)
    """
    _ = model.eval()
    # We will want one element per sequence, so we will need to split batches
    result_list = []
    for x, yu in tqdm(dataloader):
        x, yu = x.to(device=device), yu.to(device=device)
        ab = model(x)  # (batch, timesteps, 2) with padding
        yu, slens = pad_packed_sequence(yu, batch_first=True, padding_value=0)

        list_ab = unbatch_padded(ab, slens)
        list_yu = unbatch_padded(yu, slens)

        if final_only:
            list_ab = [x[-1, :] for x in list_ab]
            list_yu = [x[-1, :] for x in list_yu]
        
        result_list += [t for t in zip(list_ab, list_yu)]
        
    if to_dataframe:
        df = pd.concat([
            pd.concat([
                pd.DataFrame(ab, columns=['alpha','beta']),
                pd.DataFrame(yu, columns=['rul','uncensored'])
            ], axis=1).assign(sequence=seqnum)        
            for seqnum, (ab, yu) in enumerate(result_list)
        ], axis=0, ignore_index=True)
        return df
    else:
        return result_list