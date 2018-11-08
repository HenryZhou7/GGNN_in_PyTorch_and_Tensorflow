''' related classes for building GGNN
'''
import init_path

# torch import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data_utils
import torchvision.transforms as transforms


def mlp_decoder(mlp):
    ''' a string specifying the architecture of the mlp
    separated by ',' eg: 100,10,10
    '''
    layer_info = [int(x) for x in mlp.split(',')]
    return layer_info


class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class custom_mlp(nn.Module):

    def __init__(self, in_size, mlp_layer, nonlinear='relu'):
        super(custom_mlp, self).__init__()
        
        layer_info = [in_size] + mlp_layer
        self.mlp = nn.ModuleList()
        for i in range( len(layer_info)-1 ):
            self.mlp.append( nn.Linear(layer_info[i], layer_info[i+1]) )
        
        if nonlinear == 'relu':
            self.nonlinear = nn.LeakyReLU(0.1)
        elif nonlinear == 'tanh':
            self.nonlinear = nn.Tanh()
        elif nonlinear == 'sigmoid':
            self.nonlinear = nn.Sigmoid()
        

    def forward(self, x):
        for i, fc_layer in enumerate(self.mlp):
            x = fc_layer(x) 
            if i < len(self.mlp) - 1: 
                x = self.nonlinear(x)
        return x


class Propagator(nn.Module):

    def __init__(self, args,
                 node_state_dim, n_node,
                 n_edge_type):
        super(Propagator, self).__init__()

        # basic parameters
        self.args = args
        self.node_state_dim = node_state_dim
        self.n_node = n_node
        self.n_edge_type = n_edge_type

        self.reset_gate = nn.Sequential(
            nn.Linear(node_state_dim*3, node_state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(node_state_dim*3, node_state_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(node_state_dim*3, node_state_dim),
            nn.Tanh()
        )

    def forward(self, in_states, out_states, cur_state, adj_mat):
        ''' input:
                in_states/out_states: bs x (n_node * n_edge_type) x state_dim
                cur_state: bs x n_node x 
                           the current hidden state of each node
        '''
        self.n_node = cur_state.shape[1]
        # slice the adjacency matrix for in and out respectively
        # eg. A_in: bs x n_node x (n_node*n_edge_type)
        #           the connection adjacency matrix for all in-coming edges
        A_in  = adj_mat[:,:, :self.n_node*self.n_edge_type]
        A_out = adj_mat[:,:, self.n_node*self.n_edge_type:]

        # similar to a mask, filter out those are actually needed
        a_in  = torch.bmm(A_in, in_states)
        a_out = torch.bmm(A_out, out_states)
        a = torch.cat( (a_in, a_out, cur_state), 2 )

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joint_h = torch.cat( (a_in, a_out, r * cur_state), 2)
        h_hat = self.transform( joint_h )

        output = (1 - z) * cur_state + z * h_hat

        return output


class GGNN(nn.Module):

    def __init__(self, args, 
                 n_node, n_node_type, 
                 node_anno_dim, node_state_dim,
                 n_edge_type,
                 t_step,
                 output_dim):
        super(GGNN, self).__init__()
        assert node_state_dim >= node_anno_dim, \
            'state dim: %d anno dim: %d invalid' % \
            (node_state_dim, node_anno_dim)

        # basic parameters for GGNN
        self.args = args
        self.n_node = n_node
        self.n_node_type = n_node_type # using different embedding method

        # 
        self.node_anno_dim = node_anno_dim   # the x
        self.node_state_dim = node_state_dim # the h
        self.n_edge_type = n_edge_type
        self.t_step = t_step
        self.output_dim = output_dim

        # prepare embedding network based on different node type
        if args.embed_model is not None:
            for i in range(self.n_node_type):
                embed = custom_mlp(self.node_anno_dim,
                                   mlp_decoder(self.args.embed_model) + \
                                   [self.node_state_dim])
                self.add_module('node_type_embed_{}'.format(i), embed)
            self.embed_fcs = AttrProxy(self, 'node_type_embed_')
        
        # prepare different edge propagation weights
        for i in range(self.n_edge_type):
            if self.args.prop_model is not None:
                in_fc = custom_mlp(self.node_state_dim,
                                   mlp_decoder(self.args.prop_model) + \
                                   [self.node_state_dim])
                out_fc = custom_mlp(self.node_state_dim,
                                    mlp_decoder(self.args.prop_model) + \
                                    [self.node_state_dim])
            else:
                in_fc = custom_mlp(self.node_state_dim,
                                   [self.node_state_dim])
                out_fc = custom_mlp(self.node_state_dim,
                                    [self.node_state_dim])
            self.add_module('in_node_{}'.format(i), in_fc)
            self.add_module('out_node_{}'.format(i), out_fc)
        self.in_fcs = AttrProxy(self, 'in_node_')
        self.out_fcs = AttrProxy(self, 'out_node_')

        # prepare the propagator model
        self.propagator = Propagator(args,
            self.node_state_dim, self.n_node,
            self.n_edge_type
        )

        # prepare the output model
        self.output = custom_mlp(self.node_state_dim + self.node_anno_dim,
                                 mlp_decoder(self.args.out_model) + \
                                 [self.output_dim],
                                 nonlinear='sigmoid')

        self._initialization()
        return

    def _initialization(self):
        ''' set all the MLPs' bias to be 0 
        '''
        for m in self.modules():
            if isinstance(m, nn.ModuleList):
                for mm in m: 
                    if isinstance(mm, nn.ModuleList):
                        mm.weight.data.normal_(0.0, 0.02)
                        mm.bias.data.fill_(0)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)


    def forward(self, prop_state, annotation, adj_mat):  
        ''' input:
                prop_state: bs x n_node x state_dim the hidden state being 
                            initialized
                            the initialization
                annotation: bs x n_node x anno_dim
                            the initial information of each node, might go 
                            through mlps for the embedding
                adj_mat:    bs x n_node x 
                            ( n_node * n_edge_type * 2 in/out connection )
                            describes how the connection is set up wrt each
                            edge type
        '''
        self.n_node = prop_state.shape[1]
        # prepare annotation
        if self.args.embed_model is not None:
            # use a single embed model for now
            # to enable different node type for different embedding functions
            # need to pass in a node type mask
            prop_state = self.embed_fcs[0](annotation)


        # propagate message between nodes
        for i_step in range(self.t_step):
            in_states  = []
            out_states = []

            for i in range(self.n_edge_type):
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))

            in_states = torch.stack( tuple(in_states) ).transpose(0, 1).contiguous()
            in_states = in_states.view(
                -1, self.n_node*self.n_edge_type, self.node_state_dim
            )
            out_states = torch.stack( tuple(out_states) ).transpose(0, 1).contiguous()
            out_states = out_states.view(
                -1, self.n_node*self.n_edge_type, self.node_state_dim
            )

            prop_state = self.propagator(
                in_states, out_states, prop_state, adj_mat
            )
            pass

        # compute the output by the gnn
        all_h = torch.cat( (prop_state, annotation), 2 )
        output = self.output(all_h)
        output = output.sum(2)
        return output


if __name__ == '__main__':
    from config import config
    from data import data
    args = config.get_config()

    bs = args.bs
    n_node = args.graph_size
    n_node_type = args.node_type
    node_anno_dim = 7
    node_state_dim = 8
    n_edge_type = args.edge_type
    t_step = args.prop_ts
    output_dim = args.output_size

    model = GGNN(args, 
        n_node, n_node_type,
        node_anno_dim, node_state_dim, 
        n_edge_type, 
        t_step,
        output_dim
    )
    print(model)
    rand_state = Variable( torch.rand(bs, n_node, node_state_dim) )
    annotation = Variable( torch.rand(bs, n_node, node_anno_dim) )
    sparse_mat = data.compact2sparse_representation(data.Graph(args).mat,
        n_edge_type) 
    sparse_mat = Variable( torch.stack(
                            [torch.Tensor(sparse_mat) for i in range(bs)]
                         ) )

    y = model(rand_state, annotation, sparse_mat)
    res = y.sum(2)
    print(res.mean())
