import argparse, pickle, os
import torch.nn as nn
#from torch import tensor, save, add, rand, floor, zeros, long
import torch
import torch.optim as optim
import numpy as np

from .main import MainLinear, MainDeep


def train_linear(model):
    training_size = 100
    inputs = torch.rand(training_size, 2)  # trains model with size of training_size with dimension 2
    
    labels = torch.zeros(training_size, 2, dtype=torch.float) # create output of same size

    labels[:,1] = torch.floor(inputs[:,0]**2 + inputs[:,1]**2)  # generates output of tensor 
    labels[:,0] = torch.add(-labels[:,1], 1) 

#    labels[:,0] = floor(inputs[:,0]**2 + inputs[:,1]**2)  # generates output of tensor 
#    labels[:,1] = -labels[:,0] + 1

    model = MainLinear()
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    running_loss = 0.0

    epochs = 10
    for ep in range(epochs):
        for i in range(0, 100):
            model.train()
            optimizer.zero_grad()
#            print("session 1")
#            print(inputs, labels)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
#            print(loss)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print('Epoch %d, loss:%.4f' % (ep+1, running_loss/100))
        running_loss = 0
    print('Model State --->')
    print(model.state_dict())

    # Save the trained model
    dirname = os.path.dirname(os.path.abspath(__file__)) # Do NOT modify this line
    torch.save(model.state_dict(), os.path.join(dirname, 'linear')) # Do NOT modify this line

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
