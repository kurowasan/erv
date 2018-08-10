from torch.utils.data import Dataset, DataLoader, sampler
import os
import numpy as np
import json
import ipdb

### TODO: normalize data?

class PeptideLoader(Dataset):
    # charge is -1 for acidic, 0 for non-charged and 1 for basic
    # size is molecular weight in Daltons
    charge = {'A':0, 'R':1,'N':0, 'D':-1, 'C':0, 'Q':0, 'E':-1, 'G':0, 'H':0,
              'I':0, 'L':0, 'K':1, 'M':0, 'F':0, 'P':0, 'S':0, 'T':0,'W':0, 'Y':0}
    pka = {'C':8.3,   'D':3.9,   'E':4.3,  'H':6.0, 'K': 10.5, 'R': 12.5, 'Y':10.1}
    dalton_weight = {'A':89, 'R':174, 'N':132, 'D':133, 'C':121, 'Q':146, 'E':147, 'G':75, 'H':155,
                     'I':131, 'L':131, 'K':146, 'M':149, 'F':165, 'P':115, 'S':105, 'T':119,'W':204, 'Y':181}
    all_aa = ['C','S','T','P','A','G','N','D','E','Q','H','R','K','M','I','L','V','F','Y','W']

    def __init__(self, hparam):
        self.hparam = hparam
        self.y = np.loadtxt(hparam['data_root'] + hparam['file_y'])

        with open(os.path.join(hparam['data_root'], hparam['blosum_file'])) as f:
            self.blosum = json.load(f)
        with open(os.path.join(hparam['data_root'], hparam['file_x'])) as f:
            self.peptides = f.readlines()
        for i in range(len(self.peptides)):
            self.peptides[i] = self.peptides[i].replace('\n','').split('\t')[0]

        if self.hparam['only_aa']:
            self.size = len(self.all_aa)
        else:
            self.size = len(self.all_aa) + 3 #blosum
        self.len_peptide = 11

    def get_onehot(self, aa):
        i = self.all_aa.index(aa)
        return np.eye(len(self.all_aa))[i]

    def normalize(self, dic):
        a = np.array([val for _, val in dic.iteritems()])
        for k,v in dic.iteritems():
            dic[k] =(dic[k] - np.min(a))*1./(np.max(a) - np.min(a))
        return dic

    def convert(self, sequences):
        converted = []
        outseq = np.zeros(self.size*self.len_peptide) #*11
        for idx, aa in enumerate(sequences):
            i = idx*self.size
            if self.hparam['only_aa']:
                outseq[i:i+self.size] = self.get_onehot(aa)
            else: #blosum
                ii = len(self.all_aa)
                self.charge = self.normalize(self.charge)
                self.pka = self.normalize(self.pka)
                self.dalton_weight = self.normalize(self.dalton_weight)

                outseq[i:i+ii] = self.blosum[aa]
                outseq[i+ii] = self.charge.get(aa, 0)
                outseq[i+ii+1] = self.pka.get(aa, 0)
                outseq[i+ii+2] = self.dalton_weight.get(aa, 0)
        return outseq

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        x = self.convert(self.peptides[idx])
        y = int(round(self.y[idx]))
        return x, y, self.peptides[idx]

def convert_to_index(idx, peptides):
    new_idx = []
    non_shuffled_idx = np.array([int(i[1]) for i in peptides])
    for i in idx:
        new_idx.extend(np.where(non_shuffled_idx == i)[0])
    return new_idx

def load(hparam, with_index=True, split=True):
    if hparam['verbose']:
        print('Loading Dataset...')

    data = PeptideLoader(hparam)

    if with_index:
        with open(hparam['data_root'] + hparam['file_x']) as f:
            peptides = f.readlines()
        for i in range(len(peptides)):
            peptides[i] = peptides[i].replace('\n','').split('\t')
        idx = [int(i[1]) for i in peptides]
        idx = np.unique(idx)
        n = len(idx)
    else:
        n = len(data)
        idx = np.arange(n)

#    # if use mask
#    nb_zero = len(np.where(data.y < 0.5)[0])
#    nb_one = len(np.where(data.y > 0.5)[0])
#    idx_zero = np.random.choice(np.where(data.y < 0.5)[0], nb_zero - nb_one)
#    mask = np.ones(len(data.y), dtype=bool)
#    import ipdb;ipdb.set_trace()
#    mask[idx_zero] = False
#    idx = idx[mask]
#    np.random.shuffle(idx) # Label permutation

    if split:
        train_set = idx[:int(n*hparam['train_ratio'])]
        valid_set = idx[int(n*hparam['train_ratio']):
                        int(n*(hparam['train_ratio'] + hparam['valid_ratio']))]
        test_set = idx[int(n*(hparam['train_ratio'] + hparam['valid_ratio'])):]

        if with_index:
            train_set = convert_to_index(train_set, peptides)
            valid_set = convert_to_index(valid_set, peptides)
            test_set = convert_to_index(test_set, peptides)

        train_sampler = sampler.SubsetRandomSampler(train_set)
        valid_sampler = sampler.SubsetRandomSampler(valid_set)
        test_sampler = sampler.SubsetRandomSampler(test_set)

        train_loader = DataLoader(data, batch_size=hparam['batch_size'], sampler=train_sampler)
        valid_loader = DataLoader(data, batch_size=hparam['batch_size'], sampler=valid_sampler)
        test_loader = DataLoader(data, batch_size=hparam['batch_size'], sampler=test_sampler)

        return train_loader, valid_loader, test_loader, data.size*data.len_peptide
    else:
        valid_loader = DataLoader(data, batch_size=hparam['batch_size'])
        return valid_loader, data.size*data.len_peptide
