from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_tagger(nn.Module):
    def __init__(self, d, n_output1, n_kernel, kernels_dim):
        super(CNN_tagger, self).__init__()
        # Based on Yoon Kim article: Convolutional Neural Networks for Sentence Classification 

        self.embed_dim = d
        self.n_kernel = n_kernel
        self.kernels_dim = kernels_dim
        self.n_output1 = n_output1

        self.conv1 = nn.ModuleList([nn.Conv2d(1, n_kernel, (k, d)) for k in kernels_dim])

        self.dropout = nn.Dropout(0.25)
        self.linear1 = nn.Linear(self.n_kernel * len(self.kernels_dim), self.n_output1)

    def forward(self, x):
        x = x.unsqueeze(1)
        max_size = (max(self.kernels_dim) - 1)/2
        padding = nn.ZeroPad2d((0, 0, max_size, max_size))
        x = padding(x)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv1] #[(N,Co,W), ...]*len(Ks)
        for i, dim in enumerate(self.kernels_dim[:-1]):
            # only work for odd size...
            s = max_size - (dim - 1)/2
            x[i] = x[i][:,:,s:-s]

        logit = []
        for i in range(x[0].size(2)):
            tmp = []
            for feature_map in x:
                tmp.append(feature_map[:,:,i])
            tmp_x = torch.cat(tmp, 1)
            # x = self.dropout(x) # (N,len(Ks)*Co)
            logit.append(self.linear1(tmp_x)) # (N,C)

        return logit
