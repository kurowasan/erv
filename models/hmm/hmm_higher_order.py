from hmmlearn import hmm
import numpy as np
import torch
from torch.utils.data import dataloader, sampler

import os, sys
sys.path.append(os.path.abspath('../../utils'))
import dataset_loader

FILENAME = '../../data/viralpeptidecytotoxdataset.p'
TRAIN_RATIO = 0.9
SEQUENCE_LEN = 99
NB_CODON = 65
N_STATE = 2
N_BACK = 3
batch_size = 1
data = dataset_loader.PeptideSequence(FILENAME, False)

idx = np.arange(len(data))
np.random.seed(0)
train_data = idx[:(int)(len(data) * TRAIN_RATIO)]
test_data = idx[(int)(len(data) * TRAIN_RATIO):]
np.random.shuffle(train_data)
np.random.shuffle(test_data)
train_sampler = sampler.SubsetRandomSampler(train_data)
test_sampler = sampler.SubsetRandomSampler(test_data)
train_loader = dataloader.DataLoader(data, batch_size=batch_size, sampler=train_sampler)
test_loader = dataloader.DataLoader(data, batch_size=batch_size, sampler=test_sampler)

def train():
    n = int(len(train_data)/batch_size)
    n_entry = N_STATE**N_BACK
    pi = np.zeros(n_entry, dtype='float')
    z_last = -1
    z = np.zeros((n_entry, n_entry), dtype='float')
    n_z = np.zeros(n_entry, dtype='float')
    emission = np.zeros((n_entry, NB_CODON), dtype='float')
    emission_denom = np.zeros((n_entry, NB_CODON), dtype='float')

    for idx_batch, (x, y0, y1) in enumerate(train_loader):
        x = x.numpy().squeeze()
        y0 = y0.numpy().squeeze()

        for t in range(N_BACK, x.shape[0]):
            # convert sequence of state to int
            tmp_str = ''.join([str(i) for i in y0[t-N_BACK:t]])
            idx_y = int(tmp_str, N_STATE)
            if t == N_BACK:
                pi[idx_y] += 1

            emission[idx_y, x[t]] += 1

            if z_last > -1:
                z[z_last, idx_y] += 1

            n_z[idx_y] += 1
            z_last = idx_y
    # some states doesnt happen, remove 0 that might cause problem
    n_z[n_z == 0] = 1
    z = z/n_z[:,None]
    for i in range(z.shape[0]):
        if np.sum(z[i]) == 0:
            z[i,0] = 1

    return pi/n, z, emission/np.sum(emission, 0)

def format_data(loader):
    test_x = np.zeros((len(test_data), SEQUENCE_LEN), dtype='int')
    test_y = np.zeros((len(test_data), SEQUENCE_LEN), dtype='int')

    for i, (x, y0, y1) in enumerate(loader):
        x, y0 = x.numpy().squeeze(), y0.numpy().squeeze()
        test_x[i], test_y[i] = x, y0

    test_x = test_x.reshape(-1, 1)
    test_y = test_y.reshape(-1).astype('int')
    lengths = [SEQUENCE_LEN] * len(test_data)
    return test_x, test_y, lengths

model = hmm.MultinomialHMM(n_components=N_STATE**N_BACK)
model.startprob_, model.transmat_, model.emissionprob_ = train()
test_x, test_y, lengths = format_data(test_loader)

import ipdb;ipdb.set_trace()
z = model.predict(test_x, lengths)
correct = np.sum(z == test_y[N_BACK-1:])
