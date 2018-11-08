'''
    a tensorflow implementation of GGNN
'''
import init_path

# computing import
import numpy as np
import tensorflow as tf



class MLP(object):
    def __init__(self,
            output_size,
            scope='mlp',
            n_layers=2,
            size=10,
            activation=tf.nn.relu,
            output_activation=None
        ):
        self.output_size = output_size
        self.scope = scope
        self.n_layers = n_layers
        self.size = size
        self.activation = activation
        self.output_activation = output_activation

    def __call__(self, input, reuse=False):
        out = input
        with tf.variable_scope(self.scope, reuse=reuse):
            for _ in range(self.n_layers):
                out = tf.layers.dense(out, self.size, activation=self.activation)
            out = tf.layers.dense(out, self.output_size, activation=self.output_activation)
        return out


class ResBlock(object):
    def __init__(self,
            output_size,
            scope=3
        ):
        self.output_size

    def __call(self):
        return



class Propagator(object):
    def __init__(self,
            node_state_dim,
            n_edge_type,
            scope='propagator'
        ):
        self.node_state_dim = node_state_dim
        self.n_edge_type = n_edge_type
        self.scope = scope


        pass

    def __call__(self, 
            in_states,
            out_states,
            cur_states,
            adj_matrix, 
            reuse=False
        ):
        '''
            input:
                in_states/out_states: bs x (n_node * n_edge_type) x state_dim
                cur_states: bs x n_node x state_dim
        '''

        self.n_node = cur_states.shape[1]

        A_in  = adj_matrix[:,:, :self.n_node*self.n_edge_type]
        A_out = adj_matrix[:,:, self.n_node*self.n_edge_type:]

        with tf.variable_scope(self.scope, reuse=reuse):
            a_in = tf.matmul(A_in, in_states)
            a_out = tf.matmul(A_out, out_states)
            a = tf.concat( [a_in, a_out, cur_states], 2 )

            r = tf.layers.dense(a, self.node_state_dim, activation=tf.nn.sigmoid)
            z = tf.layers.dense(a, self.node_state_dim, activation=tf.nn.sigmoid)
            joint_h = tf.concat( [a_in, a_out, tf.multiply(r, cur_states)], 2 )
            h_hat = tf.layers.dense( joint_h, self.node_state_dim, activation=tf.nn.tanh)

            # output = (1 - z) * cur_states + z * h_hat
            output = tf.add(
                    tf.multiply(
                        tf.subtract(tf.constant(1, dtype=tf.float32), z), 
                        cur_states),
                    tf.multiply(z, h_hat)
                )

        return output



class GGNN(object):
    def __init__(self,
            n_node, n_node_type,
            node_anno_dim, 
            node_state_dim,
            node_embed_dim,
            n_edge_type,
            t_step,
            output_dim,
            ggnn_embed_layer, ggnn_embed_neuron,
            ggnn_prop_layer, ggnn_prop_neuron,
            ggnn_output_layer, ggnn_output_neuron
        ):

        self.n_node = n_node
        self.n_node_type = n_node_type

        self.node_anno_dim = node_anno_dim
        self.node_state_dim = node_state_dim
        self.node_embed_dim = node_embed_dim
        self.n_edge_type = n_edge_type
        self.t_step = t_step
        self.output_dim = output_dim

        # prepare embedding model
        self.embed_model = []
        for i in range(self.n_node_type):
            self.embed_model.append( 
                MLP(self.node_embed_dim,
                    scope='ggnn_embed_%d' % i,
                    n_layers=ggnn_embed_layer,
                    size=ggnn_embed_neuron) 
            )

        self.node_state_dim = self.node_state_dim + self.node_embed_dim

        # prepare message-passing model
        self.in_fcs = []
        self.out_fcs = []
        for i in range(self.n_edge_type):
            self.in_fcs.append(
                MLP(self.node_state_dim,
                    scope='ggnn_prop_in_%d' % i,
                    n_layers=ggnn_prop_layer,
                    size=ggnn_prop_neuron)
            )
            self.out_fcs.append(
                MLP(self.node_state_dim,
                    scope='ggnn_prop_out_%d' % i,
                    n_layers=ggnn_prop_layer,
                    size=ggnn_prop_neuron)
            )

        # prepare propagation model
        self.propagator = Propagator(
            self.node_state_dim,
            self.n_edge_type
        )

        # prepare output model
        self.output_model = MLP(
            self.output_dim,
            scope='ggnn_output',
            n_layers=ggnn_output_layer,
            size=ggnn_output_neuron
        )

        return None


    def __call__(self, prop_state, annotation, adj_mat, reuse=False):
        ''' input:
                prop_state: randomly initialized hidden states of the nodes
                annotation: meta information about the nodes should be embedded
                adj_mat:    describes how the graph connection is set up
                            separate the in and out connection type
                            size: bs x n_node x (n_node * n_edge_type * 2) 
            note:
                all inputs should be of numpy arrays
        '''
        self.bs = tf.shape(prop_state)[0]
        self.node = tf.shape(prop_state)[1]

        # prepare embedding
        # TODO: use embedding layer to process the annotation and concatenate 
        #       with the state matrix
        for i in range(self.n_node_type):
            self.embedded_feature = self.embed_model[i](
                    annotation, reuse=tf.AUTO_REUSE
                )
        prop_state = tf.concat([prop_state, self.embedded_feature], 2)

        # propagate message between different nodes
        for i_step in range(self.t_step):
            in_states = []
            out_states = []
            for i in range(self.n_edge_type):

                in_states.append(self.in_fcs[i](prop_state, reuse=tf.AUTO_REUSE))
                out_states.append(self.out_fcs[i](prop_state, reuse=tf.AUTO_REUSE))

            in_states = tf.transpose(tf.stack(in_states), perm=[1, 0, 2, 3])
            in_states = tf.reshape(in_states, 
                [-1, self.n_node*self.n_edge_type, self.node_state_dim]
            )
            out_states = tf.transpose(tf.stack(out_states), perm=[1, 0, 2, 3])
            out_states = tf.reshape(out_states,
                [-1, self.n_node*self.n_edge_type, self.node_state_dim]
            )

            prop_state = self.propagator(
                in_states, out_states, prop_state, adj_mat, reuse=tf.AUTO_REUSE
            )

            # NOTE: 
            # this line determines how to incorporate meta information into 
            # the propagation step. by default, meta info is consumed once at 
            # time step 0 
            # prop_state = tf.concat([prop_state[:,:,:self.node_state_dim], self.embedded_feature], 2)

            pass

        # compute the output based on the message
        all_h = tf.concat( [prop_state, annotation], 2 )
        output = self.output_model(all_h)
        output = tf.reshape( output, [self.bs, -1] )

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
    node_embed_dim = 12
    n_edge_type = args.edge_type
    t_step = args.prop_ts
    output_dim = args.output_size

    # build placeholder
    x_state = tf.placeholder(tf.float32, shape=(None, n_node, node_state_dim))
    x_anno  = tf.placeholder(tf.float32, shape=(None, n_node, node_anno_dim))
    x_mat   = tf.placeholder(tf.float32, shape=(None, n_node, 2 * n_node * n_edge_type))

    model = GGNN(
        n_node, n_node_type,
        node_anno_dim, node_state_dim, node_embed_dim,
        n_edge_type,
        t_step,
        output_dim,
        args.embed_layer, args.embed_neuron,
        args.prop_layer, args.prop_neuron,
        args.ggnn_output_layer, args.ggnn_output_neuron
    )

    out = model(x_state, x_anno, x_mat)

    rand_state = np.random.rand(bs, n_node, node_state_dim)
    annotation = np.random.rand(bs, n_node, node_anno_dim)
    sparse_mat = data.compact2sparse_representation(data.Graph(args).mat, n_edge_type)
    sparse_mat = np.stack([sparse_mat for i in range(bs)])

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        feed_dict = {
            x_state: rand_state,
            x_anno:  annotation,
            x_mat:   sparse_mat
        }

        output = sess.run([out], feed_dict=feed_dict)
        import pdb; pdb.set_trace()
        pass