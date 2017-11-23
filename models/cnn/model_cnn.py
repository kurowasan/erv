from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, d, n_output1, n_kernel, kernels_dim):
        super(CNN, self).__init__()
        # Based on Yoon Kim article: Convolutional Neural Networks for Sentence Classification 

        self.embed_dim = d
        self.n_kernel = n_kernel
        self.kernels_dim = kernels_dim
        self.n_output1 = n_output1

        self.conv1 = nn.ModuleList([nn.Conv2d(1, n_kernel, (k, d)) for k in
                                    kernels_dim])

        # self.conv1 = nn.Conv2d(1, 1, kernel_size=(7, d), stride=1)
        # self.conv2 = nn.Conv2d(1, 5, (7, d))
        # self.conv3 = nn.Conv2d(1, 10, (3, d))
        # self.conv4 = nn.Conv2d(1, 1, (3, d))
        # self.conv5 = nn.Conv2d(1, 1, (3, d))
        # self.conv6 = nn.Conv2d(1, 1, (3, d))

        self.dropout = nn.Dropout(0.25)
        self.linear1 = nn.Linear(self.n_kernel * len(self.kernels_dim), self.n_output1)

    def conv_and_pool(self, x):
        x = F.relu(x).squeeze(-1)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = x.unsqueeze(1)
        # x = self.conv_and_pool(self.conv1(x))
        # x = self.conv_and_pool(self.conv2(x))
        # x = self.conv_and_pool(self.conv3(x))
        # x = self.conv_and_pool(self.conv4(x))
        # x = self.conv_and_pool(self.conv5(x))
        # x = self.conv_and_pool(self.conv6(x))

        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv1] #[(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)

        x = self.dropout(x) # (N,len(Ks)*Co)
        logit = self.linear1(x) # (N,C)

        return logit, logit
