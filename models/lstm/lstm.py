from __future__ import print_function
import model_lstm

import torch
from torch.utils.data import dataloader, sampler
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.autograd import Variable
import numpy as np

import os, sys
sys.path.append(os.path.abspath('../../utils'))
import dataset_loader

N_INPUT = len(dataset_loader.PeptideSequence.all_codon) + 1
N_HIDDEN1 = 200
N_HIDDEN2 = 100
N_OUTPUT1 = 2
N_OUTPUT2 = 2

FILENAME = '../../data/viralpeptidecytotoxdataset.p'
CUDA = True
TRAIN_RATIO = 0.9
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
NB_EPOCH = 10
print_every = 10

def update_progress(progress, loss):
     sys.stdout.write('\r[{0}] {1}%   loss: {2}'.format('#'*(progress/10) + ' '*(10 - progress/10), progress, loss))

def to_categorical(x, n_classes):
    return np.eye(n_classes)[x]

dtype = torch.FloatTensor
double_type = torch.LongTensor

if CUDA:
    dtype = torch.cuda.FloatTensor
    double_type = torch.LongTensor

# Load the data and format them
data = dataset_loader.PeptideSequence(FILENAME)
idx = np.arange(len(data))
np.random.seed(0)
train_data = idx[:(int)(len(data) * TRAIN_RATIO)]
test_data = idx[(int)(len(data) * TRAIN_RATIO):]
np.random.shuffle(train_data)
np.random.shuffle(test_data)
train_sampler = sampler.SubsetRandomSampler(train_data)
test_sampler = sampler.SubsetRandomSampler(test_data)
train_loader = dataloader.DataLoader(data, batch_size=BATCH_SIZE, sampler=train_sampler)
test_loader = dataloader.DataLoader(data, batch_size=BATCH_SIZE, sampler=test_sampler)

model = model_lstm.LSTM(N_INPUT, N_HIDDEN1, N_HIDDEN2, N_OUTPUT1, N_OUTPUT2)

weight = torch.cuda.FloatTensor(2)
weight[0] = 0.1
weight[1] = 0.9

loss_function = nn.CrossEntropyLoss()
loss_function_imbalanced = nn.CrossEntropyLoss(weight)
optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)

if CUDA:
    model.cuda()

def train(epoch):
    model.train()
    n = int(len(train_data)/BATCH_SIZE)

    for idx_batch, (x, y0, y1) in enumerate(train_loader):
        x = x.type(dtype)
        if CUDA:
            x, y0, y1 = x.cuda(), y0.cuda(), y1.cuda()
        x, y0, y1 = Variable(x), Variable(y0), Variable(y1)

        optimizer.zero_grad()
        output_cyto, output_mask = model(x)
        loss = loss_function(output_cyto, y1)
        for i in range(y0.size(0)):
            weight = 1./y0.size(1) * 10.
            loss += weight * loss_function_imbalanced(output_mask[i], y0[i])
        loss.backward()
        optimizer.step()

        if idx_batch % print_every == 0:
            update_progress(int(idx_batch * 100.0/n), loss.data[0])

def test():
    model.eval()
    test_loss, mask_loss = 0, 0
    correct, mask_correct = 0, 0
    for x, y0, y1 in test_loader:
        x = x.type(dtype)
        if CUDA:
            x, y0, y1 = x.cuda(), y0.cuda(), y1.cuda()
        x, y0, y1 = Variable(x), Variable(y0), Variable(y1)
        seq_len = y0.size(0)

        output_cyto, output_mask = model(x)

        test_loss += loss_function(output_cyto, y1).data[0]
        prediction = output_cyto.data.max(1, keepdim=True)[1]
        correct += prediction.eq(y1.data.view_as(prediction)).cpu().sum()

        for i in range(seq_len):
            weight = 1./y0.size(1) * 10.
            mask_loss += weight * loss_function_imbalanced(output_mask[i], y0[i]).data[0]
            mask_prediction = output_mask[i].data.max(1, keepdim=True)[1]
            mask_correct += mask_prediction.eq(y0[i].data.view_as(mask_prediction)).cpu().sum()
    test_loss /= len(test_data)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_data),
        100. * correct / len(test_data)))
    print('\nTest set, Mask: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        mask_loss, mask_correct, len(test_data),
        100. * mask_correct / (len(test_data) * seq_len)))

for epoch in range(NB_EPOCH):
    train(epoch)
    test()
