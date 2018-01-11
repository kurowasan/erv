""" This module takes a pickle file coming from IEDB containing protein,
peptide and level of cytotoxicity. It creates a mask indicating the location of the
peptide in the protein."""
import cPickle as pickle
import numpy as np

INPUT_FILE = 'viralpeptideIEDB_dict.p' # 'peptidecytotoxdataset.p'
OUTPUT_FILE = 'viralpeptidecytotoxdataset.p'
REWEIGHT_SAMPLE = True
NB_SAMPLE = 5   # number of time to sample the same sequence
SEQUENCE_LEN = 99*3   # as SEQUENCE_LEN is in nucleo from, it should be a multiple of 3
VERBOSE = True
WILDCARD = '_'
assert SEQUENCE_LEN % 3 == 0

def codon2amino(codon):
    """ Translate a triplet of nucleotides to the corresponding amino acid """
    trad = {'TTT':'F','TTC':'F','TTA':'L','TTG':'L',
            'CTT':'L','CTC':'L','CTA':'L','CTG':'L',
            'ATT':'I','ATC':'I','ATA':'I','ATG':'M',
            'GTT':'V','GTC':'V','GTA':'V','GTG':'V',
            'TCT':'S','TCC':'S','TCA':'S','TCG':'S',
            'CCT':'P','CCC':'P','CCA':'P','CCG':'P',
            'ACT':'T','ACC':'T','ACA':'T','ACG':'T',
            'GCT':'A','GCC':'A','GCA':'A','GCG':'A',
            'TAT':'Y','TAC':'Y','TAA':'','TAG':'',
            'CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
            'AAT':'N','AAC':'N','AAA':'K','AAG':'K',
            'GAT':'D','GAC':'D','GAA':'E','GAG':'E',
            'TGT':'C','TGC':'C','TGA':'','TGG':'W',
            'CGT':'R','CGC':'R','CGA':'R','CGG':'R',
            'AGT':'S','AGC':'S','AGA':'R','AGG':'R',
            'GGT':'G','GGC':'G','GGA':'G', 'GGG':'G'}
    return trad.get(codon, WILDCARD)

def nucleo2protein(seq):
    """ Translate a sequence of nucleotides to the corresponding amino acid
    sequence"""
    protein = ''
    for i in range(0, len(seq), 3):
        if i+3 <= len(seq):
            protein += codon2amino(seq[i:i+3])
    return protein

def complement(nucleo):
    """ Return the complement of a given nucleotide """
    trad = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}
    return trad.get(nucleo, WILDCARD)

def reverseComplement(seq):
    """ Reverse and return the complement of a sequence of nucleotides"""
    seq = seq[::-1]
    newseq = ''
    for i in seq:
        newseq += complement(i)
    return newseq

def traduction(seq, peptide):
    """ Find the reading frame of the nucleotides sequence and translate it to
    its protein form accordingly """
    offset = 0
    reverse_seq = reverseComplement(seq)
    reverse = False

    if peptide in nucleo2protein(seq):
        pass
    elif peptide in nucleo2protein(seq[1:]):
        offset = 1
    elif peptide in nucleo2protein(seq[2:]):
        offset = 2
    elif peptide in nucleo2protein(reverse_seq):
        reverse = True
    elif peptide in nucleo2protein(reverse_seq[1:]):
        reverse = True
        offset = 1
    elif peptide in nucleo2protein(reverse_seq[2:]):
        reverse = True
        offset = 2
    else:
        if VERBOSE:
            print "problem with "+peptide
        return '', ''

    if reverse:
        seq = reverse_seq[offset:]
    else:
        seq = seq[offset:]
    protein = nucleo2protein(seq)

    if len(protein) < SEQUENCE_LEN:
        return '', -1
    ixs = protein.find(peptide)
    return protein, ixs

def createMask(seq, peptide, protein, ixs):
    """ Create a mask by choosing a random window around the peptide """

    # if max(0, ixs - SEQUENCE_LEN/3 + len(peptide)) <= min(ixs, len(protein) - SEQUENCE_LEN/3):
    #     # special case, if the peptide is exactly at the end
    #     start = max(0, ixs - SEQUENCE_LEN/3 + len(peptide))
    # else:
    try:
        start = np.random.randint(max(0, ixs - SEQUENCE_LEN/3 + len(peptide)),min(ixs, len(protein) - SEQUENCE_LEN/3))
    except Exception as e:
        return '', ''

    subseq = seq[start*3 : start*3 + SEQUENCE_LEN]
    mask = ('0'*((ixs-start)*3))+('1'*len(peptide)*3)
    mask += '0'*(SEQUENCE_LEN - len(mask))

    return subseq, mask

dataset = []
iterpep = 0
nb_positive, nb_negative = 0, 0
data = pickle.load(open(INPUT_FILE, 'rb'))

# createMask(data['LESEERPPTPYRNTV']['fasta'][1], 'LESEERPPTPYRNTV', 'ESSMPACNCCALPVLAGGYCRLVFHGLSYRCQRHRLGPLTSTLGSSGQLICRCTKETADTGYSSYCGCHLVHTHHLQCIMFVRLICLQFTSSAFLLSLCNLPNMEDRPTFFYSVCIAGRSWRTTRHLRLVCLCSGMQVLHMYGLARQLYFYFSSGDACAPDTSVQKEMAPFDCLWRHHVFGMCTCPHRRRCFAAESPPWSCNCGFHDAAATGFRPLALFARGPRYSWCSPFNIGSRATRVTLLAFLPHVFWTQDPCHLLLWHCRHFWAHLTLQCSFSCSYGHLVSFPFLLITCSFVIAALTWHLFYSGSPDLLFVLFMSTDQDPSGTTVPICSSTLVASLRANRRWQYFANKLQEFKQQIYTPVSICYSCLVEENKLLDCNTHFTLCSLDLFCMLLLIVAGILFILAILTEWGSGNRTYGPVFMSLGGLLTMVAGAVWLTVMTNTLLSAWILTAGFLIFLIGKCDTNRCCLVMSPFHMTYMGLAFVGFALFGVIRCCRYCCYYCLTLESEERPPTPYRNTV', 506)

# get the number of positive and negative to sample
for key in data.keys():
    iterpep += 1
    if len(data[key]['fasta']) > 1 and key != '' and (len(data[key]['fasta'][1]) > SEQUENCE_LEN+1):
        protein, ixs_peptide = (traduction(data[key]['fasta'][1], key))

        if len(protein) == 0 or ixs_peptide == -1:
            continue
        else:
            if data[key]['assay'][0] > data[key]['assay'][1]:
                nb_negative += 1
            elif data[key]['assay'][0] < data[key]['assay'][1]:
                nb_positive += 1
    if VERBOSE and iterpep % 100 == 0:
        print iterpep

# if reweight_sample is true, precalculate how many samples will be selected
# for each class
if REWEIGHT_SAMPLE:
    if nb_negative > nb_positive:
        if nb_negative < nb_positive * NB_SAMPLE:
            times, remainder = (int)(nb_negative * NB_SAMPLE/nb_positive), (nb_negative * NB_SAMPLE) % nb_positive
        else:
            times, remainder = (int)(nb_positive * NB_SAMPLE/nb_negative), (nb_positive * NB_SAMPLE) % nb_negative

        sample_positive = np.ones(nb_positive, dtype=np.int) * times + np.asarray([1]*remainder + [0]*(nb_positive - remainder))
        sample_negative = np.ones(nb_negative, dtype=np.int) * NB_SAMPLE
    else:
        if nb_positive < nb_negative * NB_SAMPLE:
            times, remainder = (int)(nb_positive * NB_SAMPLE/nb_negative), (nb_positive * NB_SAMPLE) % nb_negative
        else:
            times, remainder = (int)(nb_negative * NB_SAMPLE/nb_positive), (nb_negative * NB_SAMPLE) % nb_positive

        sample_negative = np.ones(nb_negative, dtype=np.int) * times + np.asarray([1]*remainder + [0]*(nb_negative - remainder))
        sample_positive = np.ones(nb_positive, dtype=np.int) * NB_SAMPLE

    np.random.shuffle(sample_positive)
    np.random.shuffle(sample_negative)

# sample according to the precalculated arrays and save the result
ix_positive, ix_negative = 0, 0
for key in data.keys():
    iterpep += 1
    if len(data[key]['fasta']) > 1 and key != '' and (len(data[key]['fasta'][1]) > SEQUENCE_LEN+1):
        protein, ixs_peptide = traduction(data[key]['fasta'][1], key)

        if len(protein) == 0 or ixs_peptide == -1:
            continue
        else:
            if data[key]['assay'][0] > data[key]['assay'][1]:
                cytotox = 'Negative'
                for _ in range(sample_negative[ix_negative]):
                    seq, mask = createMask(data[key]['fasta'][1], key, protein, ixs_peptide)
                    if len(seq) > 0 and len(mask) > 0:
                        dataset.append([seq, mask, cytotox])
                ix_negative += 1
            elif data[key]['assay'][0] < data[key]['assay'][1]:
                cytotox = 'Positive'
                for _ in range(sample_positive[ix_positive]):
                    seq, mask = createMask(data[key]['fasta'][1], key, protein, ixs_peptide)
                    if len(seq) > 0 and len(mask) > 0:
                        dataset.append([seq, mask, cytotox])
                ix_positive += 1
            else:
                cytotox = 'Unknown'
                seq, mask = createMask(data[key]['fasta'][1], key, protein, ixs_peptide)
                dataset.append([seq, mask, cytotox])

    if VERBOSE and iterpep%100 == 0:
        print iterpep
        pickle.dump(dataset, open(OUTPUT_FILE, 'wb'))

pickle.dump(dataset, open(OUTPUT_FILE, 'wb'))
