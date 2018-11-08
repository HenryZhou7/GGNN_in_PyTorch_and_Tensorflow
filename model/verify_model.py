''' related classes for building GGNN
'''
import init_path

# compute 
import numpy as np

# torch import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data_utils
import torchvision.transforms as transforms

# local imports
from model import model
from util import util

class verifyGraphModel(nn.Module):

    def __init__(self, args, 
                 n_node, n_node_type,
                 node_anno_dim, node_state_dim,
                 n_edge_type,
                 t_step,
                 output_dim):
        super(verifyGraphModel, self).__init__()

        self.args = args
        self.n_node = n_node
        self.n_node_type = n_node_type # using different embedding method
        self.node_anno_dim = node_anno_dim   # the x
        self.node_state_dim = node_state_dim # the h
        self.n_edge_type = n_edge_type
        self.t_step = t_step
        self.output_dim = output_dim

        self.ggnn = model.GGNN(self.args,
                               self.n_node, self.n_node_type,
                               self.node_anno_dim, self.node_state_dim,
                               self.n_edge_type,
                               self.t_step,
                               self.output_dim)


    def forward(self, prop_state, annotation, adj_mat):
        ''' input:
                prop_state: <bs> x 2 x n_node x state_dim
                annotation: <bs> x 2 x n_node x anno_dim
            output:
                predict: <bs> x 1 
                         whether the 2 graphs are isomorphic or not
        '''
        bs = prop_state.shape[0]
        bs = 2 * bs
        
        self.n_node = prop_state.shape[-2]
        prop_state = prop_state.view(-1, self.n_node, self.node_state_dim)
        annotation = annotation.view(-1, self.n_node, self.node_anno_dim)
        adj_mat = adj_mat.view(-1, self.n_node, 2*self.n_node*self.n_edge_type)

        # graph features is <bs * 2> x <output_dim>
        graph_features = self.ggnn(prop_state, annotation, adj_mat)
        # graph_features = graph_features.view(bs, 2, -1)
        # use the simplest mlp to make prediction

        # true base
        type1_index = torch.LongTensor( np.array(range(bs)) * 2 )
        type2_index = torch.LongTensor( np.array(range(bs)) * 2 + 1 )
        type1_index, type2_index = \
            util.check_gpu(self.args.gpu, type1_index, type2_index)
        type1_feature = torch.index_select(graph_features, 0, type1_index)
        type2_feature = torch.index_select(graph_features, 0, type2_index)
        diff = type1_feature - type2_feature
        dist = torch.sum( torch.mul(diff, diff), dim=1)
        return dist

if __name__ == '__main__':
    pass