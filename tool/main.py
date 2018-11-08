# system imports
import init_path
import sys
import os

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
from config import config
from data import data

from util import logger
from util import setup
from util import train
from util import visdom_util


if __name__ == '__main__':

    args = config.get_config()

    # set up visdom
    visdom_util.visdom_initialize(args)

    # set up dataset loader
    trn_loader, val_loader, tst_loader = setup.setup_dataset(args)

    # set up model
    model = setup.setup_model(args)

    # set up criterions
    crit = setup.setup_criterion(args)

    # set up optimizer
    optimizer = setup.setup_optimizer(model, args)

    # go to the correct train loop based on the task
    train.train_loop(model, trn_loader, val_loader, crit, optimizer, args)