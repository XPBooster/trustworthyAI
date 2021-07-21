#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" read datasets from existing files"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

class DataGenerator(object):


    def __init__(self, config):

        self.inputdata = pd.read_csv(config.data_path)
        self.datasize, self.d = self.inputdata.shape

        if config.normalize:
            self.inputdata = StandardScaler().fit_transform(self.inputdata)

        if config.graph_path is None:
            gtrue = np.zeros(self.d)
        else:
            gtrue = np.array(pd.read_csv(config.graph_path))
            if config.transpose:
                gtrue = np.transpose(gtrue)

        # (i,j)=1 => node i -> node j
        self.true_graph = np.int32(np.abs(gtrue) > 1e-3)

    def gen_instance_graph(self, num_nodes, dimension, test_mode=False):
        seq = np.random.randint(self.datasize, size=(dimension))
        input_ = self.inputdata[seq]
        return input_.T

    # Generate random batch for training procedure
    def train_batch(self, batch_size, num_nodes, dimension):
        input_batch = []

        for _ in range(batch_size):
            input_ = self.gen_instance_graph(num_nodes, dimension) # (feature_num, dimension(sample_seq))
            input_batch.append(input_)

        return input_batch


class CausalDataSet(Dataset):


    def __init__(self, config):

        self.config = config
        self.inputdata = pd.read_csv(config.data_path)
        self.inputdata = StandardScaler().fit_transform(self.inputdata) if config.normalize else self.inputdata
        self.datasize, self.d = self.inputdata.shape

        if config.graph_path is None:
            gtrue = np.zeros(self.d)
        else:
            gtrue = np.array(pd.read_csv(config.graph_path, index_col=0))
            if config.transpose:
                gtrue = np.transpose(gtrue)

        # (i,j)=1 => node i -> node j
        self.true_graph = np.int32(np.abs(gtrue) > 1e-3)

    def __getitem__(self, index):
        """

        Parameters
        ----------
        index: It has no meaning in this DataSet. We randomly sample data in each iter.

        Returns: Randomly sampled data.
        -------

        """
        seq = np.random.randint(self.datasize, size=(self.config.input_dimension))
        return self.inputdata[seq].T

    def __len__(self):

        return self.datasize
