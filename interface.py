import os, sys
import json
import getpass
import argparse

def parse():
    mila = False

    if mila:
        data_root = '/Tmp/{}/data/'.format(getpass.getuser())
        output = '/data/milatmp1/{}/erv/'.format(getpass.getuser())
    else:
        data_root = 'data/'
        output = 'exp/'

    parser = argparse.ArgumentParser(description='Cytotoxicity classifier')
    # arguments related to the training
    parser.add_argument('--n-epochs', type=int, default=20, help='number of epochs to train (default: 20)')
    parser.add_argument('--batch-size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument('--n-hidden', nargs='+', type=int, default=[100, 1], help='list of number of hidden units (default: 100,1)')
    parser.add_argument('--only-aa', action='store_true', help='if specified, does not use blosum and kpa, etc')
    # arguments related to the files path and other
    parser.add_argument('--data-root', default=data_root, help='Relative path to the input dataset')
    parser.add_argument('--file-x', default='new_input.txt', help='Filename of the input dataset')
    parser.add_argument('--file-y', default='new_targets.txt', help='Filename of the target dataset')
    parser.add_argument('--output', default=output, help='Relative path where the result will be logged')
    parser.add_argument('--load-config', default='n', help='Load configuration from a json file located in the output path')
    parser.add_argument('--verbose', action='store_false', help='Print messages if active (active by default)')
    parser.add_argument('--save-model', default='', help='name of the file for the model parameters. If not specified do not save the parameters')
    # parser.add_argument('--resume', default='n', help='resume the training and load the model parameters')
    args = parser.parse_args()
    hparam = vars(args)

    # if there is no existing config file,
    # create path and a new config file
    if hparam['load_config'] != 'yes':
        if not os.path.exists(hparam['output']):
            os.makedirs(hparam['output'])
        with open(os.path.join(hparam['output'], 'config.json'), 'w') as f:
            json.dump(hparam, f)
    # if load_config is chosen, then overwrite the arguments
    # with the one in the json file
    else:
        with open(os.path.join(hparam['output'], 'config.json')) as config_file:
            config = json.load(config_file)
        for key, value in hparam.items():
            if key in config:
                hparam[key] = config[key]

    # non-configurable parameters
    hparam['train_ratio'] = 0.8
    hparam['valid_ratio'] = 0.1
    hparam['cuda'] = True
    hparam['print_every'] = 10
    hparam['blosum_file'] = 'blosum62.json'
    hparam['roc_train'] = os.path.join(hparam['output'], 'roc_train.txt')
    hparam['roc_valid'] = os.path.join(hparam['output'], 'roc_valid.txt')

    hparam['curve_loss_train'] = os.path.join(hparam['output'], 'curve_loss_train.txt')
    hparam['curve_accuracy_train'] = os.path.join(hparam['output'], 'curve_accuracy_train.txt')
    hparam['curve_loss_valid'] = os.path.join(hparam['output'], 'curve_loss_valid.txt')
    hparam['curve_accuracy_valid'] = os.path.join(hparam['output'], 'curve_accuracy_valid.txt')
    hparam['log_curve'] = True

    return hparam
