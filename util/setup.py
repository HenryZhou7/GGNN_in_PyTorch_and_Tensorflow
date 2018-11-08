'''
    set up preliminaries for the training
    1. dataset
    2. model
    3. criterion
    4. optimizer

    @author: henry zhou - May 29th, 2018
'''
# compute
import numpy as np
import random

# torch import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data_utils
import torchvision.transforms as transforms

# local imports
from util import util
from util import logger
from data import data
from model import verify_model

def setup_dataset(args):
    ''' there will be different tasks
    '''
    trn_loader = data.get_dloader(args, args.dataset_size)
    val_loader = data.get_dloader(args, args.val_dataset_size, val_set=True)
    tst_loader = data.get_dloader(args, args.val_dataset_size, val_set=True)

    return trn_loader, val_loader, tst_loader

def setup_model(args):
    ''' variations of the model to come
    '''
    model = verify_model.verifyGraphModel(
        args,
        args.graph_size, args.node_type,
        args.anno_dim, args.state_dim,
        args.edge_type,
        args.prop_ts,
        args.output_size
    )

    if args.gpu != None:
        model = model.cuda(options.gpu)

    return model

def setup_criterion(args):
    '''
    '''

    return nn.BCELoss()

def setup_optimizer(model, args):
    ''' choosing the optimizer
    '''
    optimizer = util.choose_optimizer(model, args)
    return optimizer

