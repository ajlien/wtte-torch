# Subclasses of torch.utils.data.Dataset for commonly used example data

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pack_padded_sequence
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from pathlib import Path
import pandas as pd
import numpy as np
import re


class TurbofanDegradationDataset(Dataset):
    """Dataset of run-to-failure data from simulated engine systems using C-MAPSS.
    Reference: A. Saxena and K. Goebel (2008). 
    "Turbofan Engine Degradation Simulation Data Set", 
    https://ti.arc.nasa.gov/c/13/, NASA Ames, Moffett Field, CA.
    """

    # Identify which unit IDs are represented in provided files in the directory
    _regex = re.compile(r'^.*train_FD0*(\d+).txt$')

    def __init__(self, directory, train=True, unit_ids=None, min_seq_len=5, max_seq_len=100):
        self.directory = Path(directory)
        self.train = train
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.sensor_cols = ['sensor_{:02d}'.format(j+1) for j in range(21)]
        self.features = ['os_1','os_2','os_3'] + self.sensor_cols
        self.idvars = ['unit_id','run_id','cycle_num']
        self.labels = ['rul','uncensored']
        if unit_ids is None:
            unit_ids = [int(self._regex.match(str(x)).group(1))
                        for x in self.directory.glob('*.txt') 
                        if self._regex.match(str(x))]
        self.df = pd.concat([self.load_unit(i, self.train) for i in unit_ids]) \
            .reset_index(drop=True)
        # Set up feature standardizer and fit
        self.make_standardizer()
        if train:
            self.standardize(self)
        # Get valid sequence endpoints for examples
        self.valid_sequence_ends = self.determine_valid_sequence_ends()

    def __len__(self):
        return self.valid_sequence_ends.shape[0]

    def __getitem__(self, i):
        """Note: tensors returned may represent variable length sequences;
           make sure padding is performed in the corresponding DataLoader"""
        irow = self.valid_sequence_ends[i,:]
        vals = self.df.loc[irow[0]:irow[1], self.labels + self.features].values
        x  = torch.from_numpy(vals[: ,2:]).float()
        yu = torch.from_numpy(vals[: ,:2]).float()
        return x, yu

    def load_unit(self, i, train=True):
        """Create a table of data for a single turbine unit.
        :param i: Int turbine ID
        :param train: Boolean takes True if this will be training data, else test data

        It appears that every training sequence is guaranteed to end in an event (failure), 
        however the test sequences do not. The time to event corresponding to the last
        observation in each test sequence appears to be given in the RUL_FD*.txt files.
        """
        if train:
            filename_features = 'train_FD{:03d}.txt'.format(i)
        else:
            filename_features = 'test_FD{:03d}.txt'.format(i)

        df = pd.read_csv(self.directory.joinpath(filename_features),
                         sep=' ', index_col=False, header=None, usecols=[x for x in range(26)],
                         names=['run_id','cycle_num'] + self.features)
        df['unit_id'] = i
        # RUL is cycles until end of sequence for training data, but must be offset further for test data
        df['rul'] = df \
            .groupby(['unit_id','run_id']) \
            ['cycle_num'] \
            .transform(max) - df['cycle_num'] + 1

        if train:
            df['uncensored'] = 1
        else:
            # Add the end-of-sequence true RUL to the test data RUL values to offset them
            df_rul = pd.read_csv(self.directory.joinpath('RUL_FD{:03d}.txt'.format(i)),
                                 sep=' ', index_col=False, header=None, usecols=[0], names=['rul_offset'])
            df_rul['run_id'] = df_rul.index.values + 1
            df = pd.merge(df, df_rul, how='inner', on='run_id')
            df['rul'] = df['rul'] + df['rul_offset'] - 1
            df['uncensored'] = 0

        return df[self.idvars + self.labels + self.features]  # Reorder

    def make_standardizer(self):
        """Initialize a pre-processing pipeline for feature normalization that can be applied
        to other test data"""
        self.standardizer = Pipeline([('scaler', MinMaxScaler(feature_range=(-1,1))),
                                      ('remove_constant', VarianceThreshold())])
        self.standardizer.fit(self.df[self.features])
    
    def standardize(self, other):
        """Use another TurbofanDegradationDataset's standardizer to standardize this dataset's data"""
        kept_features = other.standardizer.named_steps['remove_constant'].get_support()
        kept_features = [self.features[i] for i in range(len(kept_features)) if kept_features[i]]
        self.df = pd.concat([self.df[self.idvars + self.labels],
                             pd.DataFrame(other.standardizer.transform(self.df[self.features]),
                                          columns=kept_features)], axis=1)
        self.features = kept_features

    def determine_valid_sequence_ends(self):
        """Identify the records in the dataset which may be the beginning/end of a sequence
        Returns an (N x 2) numpy array where the first column is the index of the sequence start
        and the second column is the index of the sequence end, both inclusive
        """
        idx = self.df.index.values
        # Get, for each row, the index of its run's first row and that of its run's last row
        grp = self.df \
            .reset_index() \
            .groupby(['unit_id','run_id'])['index']
        run_starts = grp.transform(np.min)
        run_ends = grp.transform(np.max)
        # The valid sequence ends are those at least min_seq_len time steps after the run started
        seq_ends = idx[idx - run_starts + 1 >= self.min_seq_len]
        run_starts_for_seq_ends = run_starts[seq_ends]
        # The valid sequence beginnings are the run start if undersized, else max_seq_len before the start
        seq_starts = np.where(seq_ends - self.max_seq_len + 1 < run_starts_for_seq_ends,
                              run_starts_for_seq_ends,
                              seq_ends - self.max_seq_len + 1)
        return np.stack([seq_starts, seq_ends], -1)

    @staticmethod
    def collate_fn(batch):
        """Use this as the collate_fn callable for a DataLoader using this Dataset.
           It will ensure that variable-length sequences within the batch are padded to the same length.
           :param batch: List of tuples (x, yu)
        """
        x, yu = zip(*batch)
        batch_x = pack_sequence(x, enforce_sorted=False)
        batch_y = pack_sequence(yu, enforce_sorted=False)
        return batch_x, batch_y
