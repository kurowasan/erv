from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import interface
import loader
import model
import utils

def flow(hparam, loader, epoch, mode='valid'): #mask=None
    n = len(loader.sampler)
    train_loss = 0
    nb_correct = 0
    curve_loss = []
    curve_accuracy = []
    valid_y, valid_peptide, valid_output = [], [], []
    embeddings = []

    model.eval()

    print('Epoch {}'.format(epoch))

    for idx_batch, (x, y, name) in enumerate(loader):
        x = x.type(dtype)
        y = y.type(dtype)
        # x[:, mask] = torch.zeros(x.size(0))
        x, y = Variable(x), Variable(y)

        output, embedding = model(x)
        loss = loss_function(output.squeeze(1), y)
        train_loss += loss.item()
        nb_correct += utils.get_nb_correct(output, y)

        utils.print_progress(hparam, idx_batch, len(loader), loss.item())
        valid_y.extend(y.cpu().numpy())
        embeddings.extend(embedding.data.cpu().numpy())
        valid_peptide.extend(name)
        valid_output.extend(output.squeeze(1).data.cpu().numpy())
        if hparam['log_curve'] and idx_batch % 100 == 0:
            curve_loss.append(loss.item())
            curve_accuracy.append(utils.get_nb_correct(output, y)*1./x.shape[0])

    utils.save_log_roc(hparam, hparam['roc_valid'], valid_peptide, valid_y, valid_output, embeddings)
    utils.save_log_curve(hparam, mode, curve_loss, curve_accuracy)

    accuracy_msg = '\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        mode, train_loss * 1./n, nb_correct, n, 100.*nb_correct/n)
    print("using the mask #{}".format(mask))
    print(accuracy_msg)
    # return nb_correct


if __name__ == '__main__':
    # parse the cli arguments
    hparam = interface.parse()

    # load the data and split them
    _, valid_loader, _, n_input = loader.load(hparam)

    # generate the model
    print('Loading Model')
    load_path = hparam['output'] + '/' + hparam['save_model']
    model = model.NonSequentialMLP([n_input] + hparam['n_hidden'])
    model.load_state_dict(torch.load(load_path))

    loss_function = nn.BCELoss()

    # if cuda is used, make sure to move model and tensors on GPU
    if hparam['cuda']:
        dtype = torch.cuda.FloatTensor
        double_type = torch.cuda.LongTensor
        model.cuda()
        if hparam['verbose']:
            print('Training on GPU')
    else:
        dtype = torch.FloatTensor
        double_type = torch.LongTensor

    # get prediction for ROC curves
    #nb_features = 253
    #pred_correct = np.zeros(nb_features)
    #for mask in range(nb_features):
    flow(hparam, valid_loader, 1, 'valid')
