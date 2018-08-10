import pandas as pd
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def update_progress(progress, loss):
     sys.stdout.write('\r[{0}] {1}%   loss: {2}'.format(
         '#'*(progress/10) + ' '*(10 - progress/10), progress, loss)
     )

def print_progress(hparam, idx_batch, n_batch, loss):
    if idx_batch % hparam['print_every'] == 0:
        update_progress(int(idx_batch * 100.0/n_batch), loss)

def get_nb_correct(output, y):
    prediction = output.squeeze().round()
    correct = prediction.eq(y).cpu().sum()
    return correct.item()

def save_log_roc(hparam, filename, name, target, pred, embedding=None):
    if embedding is None:
        d = {'peptide': name, 'prediction': pred, 'target': target}
    else:
        d = {'peptide': name, 'prediction': pred, 'target': target, 'embedding': embedding}
    df = pd.DataFrame(data=d)
    df.to_csv(filename)

def save_log_curve(hparam, log_mode, curve_loss, curve_accuracy):
    if log_mode == 'train':
        with open(hparam['curve_loss_train'], 'a') as f:
            np.savetxt(f, curve_loss)
        with open(hparam['curve_accuracy_train'], 'a') as f:
            np.savetxt(f, curve_accuracy)
    elif log_mode == 'eval':
        with open(hparam['curve_loss_valid'], 'a') as f:
            np.savetxt(f, curve_loss)
        with open(hparam['curve_accuracy_valid'], 'a') as f:
            np.savetxt(f, curve_accuracy)
