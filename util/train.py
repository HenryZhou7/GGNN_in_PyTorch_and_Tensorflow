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

from util import util
from util import logger
from util import test
from util import visdom_util

def get_loss(yhat, y, args):
    ''' compute the loss
    '''
    loss = None
    y = y.view(-1, 1)
    y = y.squeeze()

    loss = y * yhat + \
          (1 - y) * -1 * torch.clamp(yhat, min=-np.inf, max=10.0)
    loss = torch.mean(loss)
    return loss

def get_acc(yhat, y, args):
    '''
    '''
    acc = None

    bs = y.shape[0]

    dist1_idx = torch.LongTensor( np.array(range(bs)) * 2 )
    dist2_idx = torch.LongTensor( np.array(range(bs)) * 2 + 1 )
    dist1 = torch.index_select(yhat, 0, dist1_idx)
    dist2 = torch.index_select(yhat, 0, dist2_idx)

    acc = (dist1 < dist2).sum().float() / (bs + 1e-7)

    return acc

def train_loop(model, trn_dloader, val_dloader, crit, optimizer, args):
    ''' the actual train loop
    '''
    ep, it = 0, 0


    trn_loss_tracker = util.AverageMeter()
    trn_acc_tracker = util.AverageMeter()    
    loss_win = None
    acc_win = None

    while ep < args.n_epoch:
        ep = ep + 1

        model.train()
        for data_point in trn_dloader:
            it = it + 1
            optimizer.zero_grad()
            
            x1, x2, x3, y = data_point
            x1, x2, x3, y = util.check_gpu(args.gpu, x1, x2, x3, y)

            yhat = model(x1, x2, x3)
            loss = get_loss(yhat, y, args)
            acc = get_acc(yhat, y, args)

            loss.backward()
            optimizer.step()

            info = 'TRN: [EP:%3d/%3d][IT:%3d]-[Loss:%.2f]-[Acc:%.2f]' % \
                    (ep, args.n_epoch, it, loss.cpu().data,
                     acc.cpu().data)
            logger.info(info)

            trn_loss_tracker.update(loss.cpu().data, len(y))
            trn_acc_tracker.update(acc.cpu().data, len(y))


        model.eval()
        val_loss, val_acc = test.run_test(model, val_dloader, crit, args)


        # visdom plotting
        loss_win = visdom_util.viz_line(ep, [trn_loss_tracker.avg, val_loss],
            viz_win=loss_win, title='loss', xlabel='epoch', 
            ylabel='loss',
            legend=['trn', 'val']
        )
        acc_win = visdom_util.viz_line(ep, [trn_acc_tracker.avg, val_acc],
            viz_win=acc_win, title='accuracy', xlabel='epoch', 
            ylabel='accuracy (ratio of D_isomorphic < D_random)',
            legend=['trn', 'val']
        )


        trn_loss_tracker.reset()
        trn_acc_tracker.reset()




