from __future__ import print_function
import argparse
import model_cnn

import torch
from torch.utils.data import dataloader, sampler
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import time

import os, sys
sys.path.append(os.path.abspath('../bayesian/'))
import bayes
sys.path.append(os.path.abspath('../../utils'))
import dataset_loader


# Settings
parser = argparse.ArgumentParser(description='Cytotoxicity classifier')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--n-epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--kernel-dim', type=str, default='3,4,5',
                    help='comma-separated kernel dimension')
parser.add_argument('--n-kernel', type=int, default=64, metavar='N',
                    help='number of filters of each type (default: 64)')
args = parser.parse_args()
args.kernel_dim = args.kernel_dim.split(",")
args.kernel_dim = [int(x) for x in args.kernel_dim]

FILENAME = '../../data/viralpeptidecytotoxdataset.p'
LOG_PATH = '../../log/'
N_INPUT = len(dataset_loader.PeptideSequence.all_codon) + 1
N_OUTPUT1 = 2
TRAIN_RATIO = 0.9
CUDA = True
PRINT_EVERY = 10

def update_progress(progress, loss):
     sys.stdout.write('\r[{0}] {1}%   loss: {2}'.format('#'*(progress/10) + ' '*(10 - progress/10), progress, loss))

def to_categorical(x, n_classes):
    return np.eye(n_classes)[x]

if CUDA:
    dtype = torch.cuda.FloatTensor
    double_type = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    double_type = torch.LongTensor

# Load the data and format them
data = dataset_loader.PeptideSequence(FILENAME)
# np.random.seed(0)

test_set_completed = 0
proteins = data.getProteinsName()
tmp_proteins = data.getProteinsName()
train_data = []
test_data = []

while(not test_set_completed):
    rand = np.random.randint(len(tmp_proteins))
    protein_chosen = tmp_proteins[rand]
    idx_train = [i for i, x in enumerate(proteins) if x == protein_chosen]
    train_data.extend(idx_train)
    tmp_proteins = [x for i, x in enumerate(tmp_proteins) if x != protein_chosen]

    rand = np.random.randint(len(tmp_proteins))
    protein_chosen = tmp_proteins[rand]
    idx_test = [i for i, x in enumerate(proteins) if x == protein_chosen]
    test_data.extend(idx_test)
    tmp_proteins = [x for i, x in enumerate(tmp_proteins) if x != protein_chosen]

    if(len(test_data) >= (1 - TRAIN_RATIO)*len(data)):
        test_set_completed = 1

for protein in set(tmp_proteins):
    for i, x in enumerate(proteins):
        if(x == protein):
            train_data.append(i)

print("Effective test dataset ratio: ")
print(len(test_data)*1./len(data))

'''
idx = np.arange(len(data))
train_data = idx[:(int)(len(data) * TRAIN_RATIO)]
test_data = idx[(int)(len(data) * TRAIN_RATIO):]
'''

np.random.shuffle(train_data)
np.random.shuffle(test_data)
train_sampler = sampler.SubsetRandomSampler(train_data)
test_sampler = sampler.SubsetRandomSampler(test_data)

train_loader = dataloader.DataLoader(data, batch_size=args.batch_size, sampler=train_sampler)
test_loader = dataloader.DataLoader(data, batch_size=args.batch_size, sampler=test_sampler)

model = model_cnn.CNN(N_INPUT, N_OUTPUT1, args.n_kernel, args.kernel_dim)


weight = torch.cuda.FloatTensor(2)

weight[0] = 0.5
weight[1] = 0.5

loss_function = nn.CrossEntropyLoss(weight)
optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

if CUDA:
    model.cuda()

def train(epoch):
    train_loss = 0
    correct = 0
    model.train()
    n = int(len(train_data)/args.batch_size)

    print('Epoch {}: '.format(epoch))

    for idx_batch, (x, y0, y1) in enumerate(train_loader):
        x = x.type(dtype)
        if CUDA:
            x, y0, y1 = x.cuda(), y0.cuda(), y1.cuda()
        x, y0, y1 = Variable(x), Variable(y0), Variable(y1)

        optimizer.zero_grad()
        output_cyto, output_mask = model(x)
        loss = loss_function(output_cyto, y1)
        train_loss += loss.data[0]
        loss.backward()
        optimizer.step()

        prediction = output_cyto.data.max(1, keepdim=True)[1]
        correct += prediction.eq(y1.data.view_as(prediction)).cpu().sum()


        if idx_batch % PRINT_EVERY == 0:
            update_progress(int(idx_batch * 100.0/n), loss.data[0])

    accuracy_msg = '\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss * 1./len(train_data), correct, len(train_data), 100. * correct / len(train_data))
    print(accuracy_msg)

def test(epoch):
    model.eval()
    test_loss, mask_loss = 0, 0
    correct, mask_correct = 0, 0
    for x, y0, y1 in test_loader:
        x = x.type(dtype)
        if CUDA:
            x, y0, y1 = x.cuda(), y0.cuda(), y1.cuda()
        x, y0, y1 = Variable(x), Variable(y0), Variable(y1)

        # if epoch == args.n_epochs-1:
        #     import pdb; pdb.set_trace()

        output_cyto, output_mask = model(x)

        test_loss += loss_function(output_cyto, y1).data[0]
        prediction = output_cyto.data.max(1, keepdim=True)[1]
        correct += prediction.eq(y1.data.view_as(prediction)).cpu().sum()

    accuracy_msg = 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss * 1./len(test_data), correct, len(test_data), 100. * correct / len(test_data))

    print(accuracy_msg)
    log(accuracy_msg)

def list2str(l):
    return ",".join(str(x) for x in l)

def log(txt):
    with open(log_file, 'a') as f:
        f.write(txt)


log_file = LOG_PATH + "CNN_" + time.strftime("%Y_%m_%d_%H_%M") + ".txt"
log("Batch Size: {} \nLearning Rate: {} \nNumber of epochs: {} \nKernel dimensions: {} \n Number of Kernel: {}\n".format(
    args.batch_size, args.lr, args.n_epochs, list2str(args.kernel_dim), args.n_kernel))


### benchmark on Bayes model
kmer_probs = bayes.prepare_tables(train_loader)
y_pred, y_true = bayes.bayes_predict(kmer_probs,test_loader)
tn, fp, fn, tp, auc, acc = bayes.calculate_statistics(y_pred, y_true)
bayes.bayes_log(tn, fp, fn, tp, auc, acc, log_file)


for epoch in range(args.n_epochs):
    train(epoch)
    test(epoch)
