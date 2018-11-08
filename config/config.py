import init_path
import os
import sys
import argparse

def get_config():
    '''
    '''
    usage = """
        Usage of this tool.
        $ python main.py [--train]
    """

    parser = argparse.ArgumentParser(description='GGNN')

    # dataset setup
    parser.add_argument('--dataset', type=str, default='vGraph')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--difficulty', type=int, default=3)
    # trn set settings
    parser.add_argument('--dataset_size', type=int, default=1024)
    parser.add_argument('--graph_size', type=int, default=5)
    # val set settings
    parser.add_argument('--val_dataset_size', type=int, default=None)
    parser.add_argument('--val_graph_size', type=int, default=None)
    
    parser.add_argument('--node_type', type=int, default=1)
    parser.add_argument('--edge_type', type=int, default=1)
    parser.add_argument('--connection_rate', type=float, default=0.1)
    parser.add_argument('--random_embed', action='store_true', default=False)
    parser.add_argument('--output_size', type=int, default=8)

    # parameters that is essential to GGNN are used when creating the dataset
    parser.add_argument('--state_dim', type=int, default=5)
    parser.add_argument('--anno_dim', type=int, default=4)

    # pytorch model setup
    parser.add_argument('--embed_model', type=str, default=None)
    parser.add_argument('--prop_model', type=str, default=None)
    parser.add_argument('--out_model', type=str, default='8')
    parser.add_argument('--prop_ts', type=int, default=5)

    # tensorflow model setup
    parser = tf_gnn_model(parser)

    # training setup
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--optimizer', default='sgd', type=str,
                        choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--n_epoch', type=int, default=500)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--nt', type=int, default=5)

    # visdom 
    parser = visdom_config(parser)

    # final
    args = parser.parse_args()
    args = post_process(args)
    
    return args

def tf_gnn_model(parser):


    parser.add_argument('--embed_layer', type=int, default=1)
    parser.add_argument('--embed_neuron', type=int, default=10)
    parser.add_argument('--prop_layer', type=int, default=1)
    parser.add_argument('--prop_neuron', type=int, default=10)
    parser.add_argument('--ggnn_output_layer', type=int, default=1)
    parser.add_argument('--ggnn_output_neuron', type=int, default=10)

    parser.add_argument('--prop_cell', type=str, default='gru')

    return parser

def visdom_config(parser):
    '''
    '''
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument('--vis_server', type=str, default='http://18.219.137.243')
    parser.add_argument('--vis_port', type=int, default=4212)

    return parser

def post_process(args):
    '''
    '''
    # process the val dataset
    if args.val_dataset_size is None:
        args.val_dataset_size = int(0.2 * args.dataset_size)

    if args.val_graph_size is None:
        args.val_graph_size = args.graph_size



    return args





if __name__ == '__main__':
    pass