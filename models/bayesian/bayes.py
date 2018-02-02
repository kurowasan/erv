from __future__ import print_function
import cPickle as pickle
import random
from random import shuffle
import numpy
from sklearn.metrics import confusion_matrix, metrics


KLENGTH = 3
KMERBASE = 'codon' #'aminoacid'
if KMERBASE == 'codon':
	KLENGTH = 3

FILENAME = '../../data/viralpeptidecytotoxdataset.p'

def convert_to_codon(sequence):
	all_codon  = ['TTT','TTC','TTA','TTG','CTT','CTC','CTA','CTG',
	            'ATT','ATC','ATA','ATG','GTT','GTC','GTA','GTG',
	            'TCT','TCC','TCA','TCG','CCT','CCC','CCA','CCG',
	            'ACT','ACC','ACA','ACG','GCT','GCC','GCA','GCG',
	            'TAT','TAC','TAA','TAG','CAT','CAC','CAA','CAG',
	            'AAT','AAC','AAA','AAG','GAT','GAC','GAA','GAG',
	            'TGT','TGC','TGA','TGG','CGT','CGC','CGA','CGG',
	            'AGT','AGC','AGA','AGG','GGT','GGC','GGA', 'GGG','XXX']

	converted = []
	for x in sequence:
		temp = ""
		for i in x:
			temp+=all_codon[numpy.argmax(i)]
		converted.append(temp)
	return converted

def prepare_tables(train_data):
	kmer_probs = {}
	kmer_probs['neg'] = {}
	kmer_probs['pos'] = {}
	types = ['neg','pos']
	

	for x, y0, y1 in train_data:
		batch_x = x.numpy()
		batch_x_codons = convert_to_codon(batch_x)
		batch_y1 = y1.numpy()

		for single_x, single_y in zip(batch_x_codons, batch_y1):

			if int(single_y)>=0.5:
				positive = True
			else:
				positive = False
			seq = single_x
			### run through the sequence by 3 nucleotides	
			for i in xrange(0,len(seq)-(KLENGTH-1)):
				codon = seq[i:i+KLENGTH]
				if positive:
					part = 1
				else:
					part = 0
				if codon not in kmer_probs[types[part]]: #chose proper dictionnary
					kmer_probs[types[part]][codon] = 1
					kmer_probs[types[part]]['total'] = 1
				else:
					kmer_probs[types[part]][codon] += 1
					kmer_probs[types[part]]['total'] += 1

	### conditionning the kmer tables to log probabilities (from counts)
	for typ in kmer_probs.keys():
		for codon in kmer_probs[typ]:
			if not codon == 'total':
				kmer_probs[typ][codon]/float(kmer_probs[typ]['total'])
				kmer_probs[typ][codon] = numpy.log10(kmer_probs[typ][codon])
	return kmer_probs

def bayes_predict(kmer_probs,test_set):
	y_pred = []
	y_true = []
	types = ['neg','pos']
	counter = 0
	for x, y0, y1 in test_set:
			batch_x = x.numpy()
			batch_x_codons = convert_to_codon(batch_x)
			batch_y1 = y1.numpy()

			for single_x, single_y in zip(batch_x_codons, batch_y1):
				counter+=1
				y_true.append(single_y)
				seq = single_x
				proba_pos = 0
				proba_neg = 0
				seq = single_x
				for i in xrange(0,len(seq)-(KLENGTH-1)):
					codon = seq[i:i+KLENGTH]
					if codon in kmer_probs['neg']:
						proba_neg+=kmer_probs['neg'][codon]
					if codon in kmer_probs['pos']:
						proba_pos+=kmer_probs['pos'][codon]
				pred = numpy.argmax([proba_neg, proba_pos])
				y_pred.append(pred)
	return y_pred,y_true

def calculate_statistics(y_pred,y_true):
	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
	print ('Bayes model results:')
	print ('True Positives: '+str(tp))
	print ('True Negative: '+str(tn))
	print ('False Positives: '+str(fp))
	print ('False Negatives: '+str(fn))
	fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
	print ('AUC ' + str(metrics.auc(fpr, tpr)))
	#return tn,fp,fn,tp,str(metrics.auc(fpr, tpr))






