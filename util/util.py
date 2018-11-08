''' generic util functions
'''
import os
import sys
import pdb
import uuid
import shlex
import itertools
import subprocess
import numpy as np
import pickle as pkl
from tqdm import tqdm
from tabulate import tabulate

# Torch imports
import torch
from torch.autograd import Variable
import torch.nn.functional as F

def choose_optimizer(model, options):
    """Select an optimizer used for training based on the options.
    """

    try:
        wd = options.weight_decay
    except:
        wd = 0.0

    if options.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=options.lr, weight_decay=wd)
    elif options.optimizer == 'mom':
        return torch.optim.SGD(model.parameters(), momentum=0.9, lr=options.lr, weight_decay=wd)
    elif options.optimizer == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=options.lr, weight_decay=wd)
    elif options.optimizer == 'adagrad':
        return torch.optim.Adagrad(model.parameters(), lr=options.lr, weight_decay=wd)
    elif options.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=options.lr, weight_decay=wd)
    return optimizer


def check_gpu(gpu, *args):
    """Move data in *args to GPU?
        gpu: options.gpu (None, or 0, 1, .. gpu index)
    """

    if gpu == None:
        if isinstance(args[0], dict):
            d = args[0]
            #print(d.keys())
            var_dict = {}
            for key in d:
                var_dict[key] = Variable(d[key])
            if len(args) > 1:
                return [var_dict] + check_gpu(gpu, *args[1:])
            else:
                return [var_dict]
        # it's a list of arguments
        if len(args) > 1:
            return [Variable(a) for a in args]
        else:  # single argument, don't make a list
            return Variable(args[0])

    else:
        if isinstance(args[0], dict):
            d = args[0]
            #print(d.keys())
            var_dict = {}
            for key in d:
                var_dict[key] = Variable(d[key].cuda(gpu))
            if len(args) > 1:
                return [var_dict] + check_gpu(gpu, *args[1:])
            else:
                return [var_dict]
        # it's a list of arguments
        if len(args) > 1:
            return [Variable(a.cuda(gpu)) for a in args]
        else:  # single argument, don't make a list
            return Variable(args[0].cuda(gpu))

class AverageMeter(object):
    """Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count + 1e-6)
