from __future__ import print_function
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class PeptideSequence(Dataset):
    """Load the peptide dataset one batch at a time, keeping the memory free"""
    all_codon  = ['TTT','TTC','TTA','TTG','CTT','CTC','CTA','CTG',
                'ATT','ATC','ATA','ATG','GTT','GTC','GTA','GTG',
                'TCT','TCC','TCA','TCG','CCT','CCC','CCA','CCG',
                'ACT','ACC','ACA','ACG','GCT','GCC','GCA','GCG',
                'TAT','TAC','TAA','TAG','CAT','CAC','CAA','CAG',
                'AAT','AAC','AAA','AAG','GAT','GAC','GAA','GAG',
                'TGT','TGC','TGA','TGG','CGT','CGC','CGA','CGG',
                'AGT','AGC','AGA','AGG','GGT','GGC','GGA', 'GGG']

    def __init__(self, filename, onehot=True):
        self.dict = pickle.load(open(filename, "rb"))
        self.sequence_len = len(self.dict[0][0])/3
        self.n_codon = len(self.all_codon) + 1
        self.onehot = onehot

    def codonToIndex(self, codon):
        try:
            index = self.all_codon.index(codon)
        except ValueError:
            index = -1
        return index

    def getProteinsName(self):
        proteins = []

        for i in range(len(self.dict)):
            proteins.append(self.dict[i][0])

        return proteins

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, idx):
        """Data structure:
            [protein_name, nucleo_seq, mask, cytotoxicity]
            encode x as onehot
        """
        seq = self.dict[idx]
        y0 = np.zeros((self.sequence_len)).astype('int')
        y1 = np.zeros(1).astype('int')

        if self.onehot:
            x = np.zeros((self.sequence_len, self.n_codon)).astype('float64')
            for i in range(self.sequence_len):
                j = i*3
                if self.codonToIndex(seq[1][j:j+3]) == -1:
                    x[i, 64] = 1.
                else:
                    x[i, self.codonToIndex(seq[1][j:j+3])] = 1.
                y0[i] = str(seq[2][j])
        else:
            x = np.zeros(self.sequence_len).astype('int')
            for i in range(self.sequence_len):
                j = i*3
                if self.codonToIndex(seq[1][j:j+3]) == -1:
                    x[i] = 64
                else:
                    x[i] = self.codonToIndex(seq[1][j:j+3])
                y0[i] = str(seq[2][j])


        if seq[3] == 'Positive':
            y1 = 1
        elif seq[3] == 'Negative':
            y1 = 0
        else:
            y1 = 0 # to remove, unknown!!!

        return x, y0, y1
