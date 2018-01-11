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

        self.dropout = nn.Dropout(0.25)
        self.linear1 = nn.Linear(self.n_kernel * len(self.kernels_dim), self.n_output1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv1] #[(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        import pdb; pdb.set_trace()
        x = torch.cat(x, 1)

        x = self.dropout(x) # (N,len(Ks)*Co)
        logit = self.linear1(x) # (N,C)

        return logit, logit
