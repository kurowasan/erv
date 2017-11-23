from __future__ import print_function
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output1, n_output2):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTMCell(n_input, n_hidden1)
        self.lstm2 = nn.LSTMCell(n_hidden1, n_hidden2)
        self.linear1 = nn.Linear(n_hidden2, n_output1)
        self.linear2 = nn.Linear(n_hidden2, n_output2)

        self.n_input = n_input
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_output1 = n_output1
        self.n_output2 = n_output2

    def forward(self, x, future=0):
        outputs = []

        h_t = Variable(torch.cuda.FloatTensor(x.size(0), self.n_hidden1).fill_(0), requires_grad=False)
        c_t = Variable(torch.cuda.FloatTensor(x.size(0), self.n_hidden1).fill_(0), requires_grad=False)
        h_t2 = Variable(torch.cuda.FloatTensor(x.size(0), self.n_hidden2).fill_(0), requires_grad=False)
        c_t2 = Variable(torch.cuda.FloatTensor(x.size(0), self.n_hidden2).fill_(0), requires_grad=False)

        for i, x_t in enumerate(x.chunk(x.size(1), dim=1)):
            h_t, c_t = self.lstm1(x_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output_mask = self.linear1(h_t2)
            outputs += [output_mask]
        for i in range(future):
            h_t, c_t = self.lstm1(h_t2, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            outputs += [h_t2]

        # outputs = torch.stack(outputs, self.n_output).squeeze(2)
        output_cyto = self.linear2(h_t2)
        outputs = torch.stack(outputs, 1)

        return output_cyto, outputs
