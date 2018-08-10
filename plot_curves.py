from sklearn.metrics import roc_curve, auc
import time
import os
import numpy as np
import getpass
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import pandas as pd

parser = argparse.ArgumentParser(description='Plot curves (loss, accuracy, roc) of trained models')
parser.add_argument('--path', default='', help='path to the file or to a directory')
parser.add_argument('--all', action='store_false', help='Plot all the files in the folder')
parser.add_argument('-m', '--mila', action='store_true', help='set to true if on mila cluster')
args = parser.parse_args()

if args.mila:
    data_root = '/data/milatmp1/{}/erv/'.format(getpass.getuser())
else:
    data_root = ''

args.path = os.path.join(data_root, args.path)

def plot_and_save(curve, name):
    plt.figure()
    plt.plot(curve)
    if 'loss' in name:
        plt.title("Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
    if 'accuracy' in name:
        plt.title("Accuracy")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
    plt.savefig(name.replace('.txt','.png'))

def get_filename(name):
    if '/' in name:
        return args.path.split('/')[-1]
    else:
        return name

if not args.all:
    curve = np.loadtxt(args.path)
    plot_and_save(curve, get_filename(args.path))
else:
    for f in glob.glob(str(args.path) + '*.txt'):
        if 'roc' not in f:
            curve = np.loadtxt(f)
            plot_and_save(curve, f)

    ### ROC curve
    plt.figure()
    y = pd.read_csv(os.path.join(str(args.path),'roc_valid.txt'))
    y_pred = y['prediction'].values
    y_test = y['target'].values

    plt.hist(y_pred[y_test==0],label='0',alpha=0.2)
    plt.hist(y_pred[y_test==1], label='1',alpha=0.2)
    plt.legend(loc='upper right')

    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()

    plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve (area = {0:.4f})'.format(float(roc_auc)))
    plt.plot([0, 1], [0, 1], color='navy', lw=0.25, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC on neural net prediction')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(str(args.path), 'roc_valid.png'))
