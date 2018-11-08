# system imports
import init_path
import sys
import os

# compute
import numpy as np
import random
import tensorflow as tf

# local import
from config import config
from data import data

from util import logger
from util import visdom

from tf_util import tf_setup
from tf_util import tf_train

if __name__ == '__main__':

    args = config.get_config()

    # set up visdom
    visdom_util.visdom_initialize(args)

    # set up dataset
    

    # set up model


    # set up optimizer and criterion


    # actutual training loop