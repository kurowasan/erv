import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

### TODO: adjustable size, nb of hidden layers
class NonSequentialMLP(nn.Module):
    def __init__(self, layer_list):
        super(NonSequentialMLP, self).__init__()
        self.layer = []
        self.nb_layers = len(layer_list) - 1
        for i in range(self.nb_layers):
            self.layer.append(nn.Linear(layer_list[i], layer_list[i+1]))
        self.dense = nn.ModuleList(self.layer)

    def forward(self, x):
        for i in range(self.nb_layers - 1):
            x = self.dense[i](x)
            if i <= 1:
                x = F.relu(x)
        out = self.dense[-1](x)
        out = F.sigmoid(out)
        return out, x

class DenseLayer(nn.Module):
    def __init__(self, n_input, n_output, non_linear='sigmoid'):
        super(DenseLayer, self).__init__()
        self.dense = nn.Sequential()
        self.dense.add_module('dense', nn.Linear(n_input, n_output))
        # self.dense.add_module('batch_norm', nn.BatchNorm1d(n_output))
        # self.dense.add_module('dropout', nn.Dropout(0.2))
        if non_linear == 'relu':
            self.dense.add_module('dropout', nn.Dropout(0.5))
            self.dense.add_module('relu', nn.ReLU())
        elif non_linear == 'tanh':
            self.dense.add_module('dropout', nn.Dropout(0.5))
            self.dense.add_module('tanh', nn.Tanh())
        elif non_linear == 'sigmoid':
            self.dense.add_module('sigmoid', nn.Sigmoid())

    def forward(self, x):
        return self.dense.forward(x)

class MLP(nn.Module):
    def __init__(self, layer_list): # n_input=23*11, n_hidden=100, n_output=1): #*11
        super(MLP, self).__init__()
        self.mlp = nn.Sequential()
        names = ['layer_{}'.format(i) for i in range(len(layer_list))]
        non_linear = ['sigmoid'] * (len(layer_list) - 1) + ['sigmoid']
        for name, h1, h2 in zip(names, layer_list[:-1], layer_list[1:]):
            self.mlp.add_module(name, DenseLayer(h1, h2))

    def forward(self, x):
        return self.mlp.forward(x)

class ConvLayer(nn.Module):
    def __init__(self, n_input, n_output, kernel_size, padding=None):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv', nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=padding))
        self.conv.add_module('norm', nn.BatchNorm2d(n_output))
        self.conv.add_module('relu', nn.ReLU())

    def forward(self, x):
        return self.conv.forward(x)

class CNN(nn.Module):
    def __init__(self, layer_list):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential()
        names = ['layer_{}'.format(i) for i in range(len(layer_list))]
        for name, h1, h2 in zip(names, layer_list[:-1], layer_list[1:]):
            self.cnn.add_module(name, DenseLayer(h1, h2))

    def forward(self, x):
        return self.cnn.forward(x)

