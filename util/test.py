# system imports
import init_path
import sys
import os
from tqdm import tqdm

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
from util import train

def get_acc(yhat, y):
    ''' for binary cross entropy measurement 
    take in 2 numpy arrays
    yhat's element has range [0, 1]
    y is either 1 or 0
    '''
    yhat[yhat >= 0.5] = 1
    yhat[yhat <  0.5] = 0
    res = np.equal(yhat, y)
    correct = res.sum()
    acc = correct / (1e-7 + res.size)
    return acc


def run_test(model, eval_loader, crit, args):
    '''
    '''
    loss_tracker = util.AverageMeter()
    acc_tracker = util.AverageMeter()

    for data_point in tqdm(eval_loader):

        x1, x2, x3, y = data_point
        x1, x2, x3, y = util.check_gpu(args.gpu, x1, x2, x3, y)

        yhat = model(x1, x2, x3)
        loss = train.get_loss(yhat, y, args)
        acc = train.get_acc(yhat, y, args)

        loss_tracker.update(float(loss.data), len(y))
        acc_tracker.update(float(acc.data),len(y))

    info = 'VAL: [Loss:%.2f]-[ACC:%.2f]' % (loss_tracker.avg, acc_tracker.avg)
    logger.info(info)
    return loss_tracker.avg, acc_tracker.avg

