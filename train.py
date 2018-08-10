from __future__ import print_function
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import interface
import loader
import model
import utils

### TODO: add a resume feature
### TODO: an automatic curve display
### TODO: add optimizers, add model ?
### TODO: add roc curve displayer

def flow(hparam, loader, epoch, mode, log_roc=False, log_mode=''):
    n = len(loader.sampler)
    train_loss = 0
    nb_correct = 0
    curve_loss = []
    curve_accuracy = []
    train_y, train_peptide, train_output = [], [], []
    valid_y, valid_peptide, valid_output = [], [], []
    x_gradient = torch.zeros((253)).cuda()

    if mode == 'train':
        model.train()
    elif mode == 'eval':
        model.eval()

    if hparam['verbose']:
        print('Mode: {}'.format(mode))
        print('Epoch {}'.format(epoch))

    for idx_batch, (x, y, name) in enumerate(loader):
        x = x.type(dtype)
        y = y.type(dtype)
        x, y = Variable(x, requires_grad=True), Variable(y)

        if mode == 'train':
            optimizer.zero_grad()
        output, _ = model(x)
        loss = loss_function(output.squeeze(1), y)
        train_loss += loss.item()
        if mode == 'train':
            loss.backward(retain_graph=True)
            optimizer.step()

        if idx_batch < 100 and mode=='get_grad' and epoch == 10:
            x_gradient += torch.sum(torch.abs(x.grad), 0)
        if idx_batch==200 and mode=='get_grad' and epoch == 10:
            print(x.grad)
            import ipdb;ipdb.set_trace()
            print(x_gradient)
            print(np.argsort(x_gradient))
            new_x = np.argsort(x_gradient)
            new_x = new_x.numpy()
            print("205 is at rank #{}".format(np.where(new_x == 205)[0]))
            print("45 is at rank #{}".format(np.where(new_x == 45)[0]))
            x_gradient = torch.zeros((253)).cuda()

        nb_correct += utils.get_nb_correct(output, y)

        if hparam['verbose']:
            utils.print_progress(hparam, idx_batch, len(loader), loss.item())
        if log_roc:
            if log_mode == 'train':
                train_y.extend(y.cpu().numpy())
                train_peptide.extend(name)
                train_output.extend(output.squeeze(1).data.cpu().numpy())
            elif log_mode == 'eval':
                valid_y.extend(y.cpu().numpy())
                valid_peptide.extend(name)
                valid_output.extend(output.squeeze(1).data.cpu().numpy())
        if hparam['log_curve'] and log_mode=='' and idx_batch % 100 == 0:
            curve_loss.append(loss.item())
            curve_accuracy.append(utils.get_nb_correct(output, y)*1./x.shape[0])

    if log_roc:
        if log_mode == 'train':
            utils.save_log_roc(hparam, hparam['roc_train'], train_peptide, train_y, train_output)
        elif log_mode == 'eval':
            utils.save_log_roc(hparam, hparam['roc_valid'], valid_peptide, valid_y, valid_output)

    if hparam['log_curve'] and log_mode=='':
        utils.save_log_curve(hparam, mode, curve_loss, curve_accuracy)

    if hparam['verbose']:
        accuracy_msg = '\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            mode, train_loss * 1./n, nb_correct, n, 100.*nb_correct/n)
        print(accuracy_msg)


if __name__ == '__main__':
    # parse the cli arguments
    hparam = interface.parse()
    torch.manual_seed(1)

    # load the data and split them
    train_loader, valid_loader, test_loader, n_input = loader.load(hparam)

    # generate the model
    if hparam['verbose']:
        print('Generating Model')
    model = model.NonSequentialMLP([n_input] + hparam['n_hidden']) #MLP
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=hparam['lr'])

    # if cuda is used, make sure to move model and tensors on GPU
    if hparam['cuda']:
        dtype = torch.cuda.FloatTensor
        double_type = torch.cuda.LongTensor
        model.cuda()
        if hparam['verbose']:
            print('Training on GPU')
            if hparam['save_model'] != '':
                print('Saving parameters in {}'.format(hparam['save_model']))
    else:
        dtype = torch.FloatTensor
        double_type = torch.LongTensor

    # train and evaluate the model
    for epoch in range(hparam['n_epochs']):
        flow(hparam, train_loader, epoch, 'train')
        flow(hparam, valid_loader, epoch, 'eval')

    # Test on self_peptides
    # hparam['file_x'] = "mice_input.txt"
    # hparam['file_y'] = "mice_targets.txt"
    # hparam['train_ratio'] = 0.9
    # hparam['valid_ratio'] = 0.05
    # train_loader, valid_loader, test_loader, n_input = loader.load(hparam, False)
    # get prediction for ROC curves
    flow(hparam, train_loader, 1, 'eval', True, 'train')
    flow(hparam, valid_loader, 1, 'eval', True, 'eval')

    if hparam['save_model'] != '':
        torch.save(model.state_dict(), hparam['output'] + '/' + hparam['save_model'] )
