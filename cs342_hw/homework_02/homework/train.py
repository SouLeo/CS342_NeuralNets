import argparse, pickle, os
import torch.nn as nn
from torch import tensor, save
import torch.optim as optim
import numpy as np

from .main import MainLinear, MainDeep


def train_linear(model):
    '''
    Your code here
    '''
    # Save the trained model
    dirname = os.path.dirname(os.path.abspath(__file__)) # Do NOT modify this line
    save(model.state_dict(), os.path.join(dirname, 'linear')) # Do NOT modify this line

def train_deep(model):
    '''
    Your code here
    '''

    # Save the trained model
    dirname = os.path.dirname(os.path.abspath(__file__)) # Do NOT modify this line
    save(model.state_dict(), os.path.join(dirname, 'deep')) # Do NOT modify this line


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['linear', 'deep'])
    args = parser.parse_args()

    if args.model == 'linear':
        print ('[I] Start training linear model')
        train_linear(MainLinear())
    elif args.model == 'deep':
        print ('[I] Start training linear model')
        train_deep(MainDeep())

    print ('[I] Training finished')
