
import init_path

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


class Propagator(object):
    def __init__(self,
            # 
            node_state_dim,
            n_edge_type,
            # placeholder
            in_states, out_states, cur_states, adj_matrix,
            scope='propagator'
        ):
        # save config
        self.node_state_dim = node_state_dim
        self.n_edge_type = n_edge_type
        self.scope = scope
        # save input
        self.in_states = in_states
        self.out_states = out_states
        self.cur_states = cur_states
        self.adj_matrix = adj_matrix

        self._build()


    def _build(self):
        self._n_node = tf.shape(self.cur_states)[1]

        A_in = self.adj_matrix[:,:, :self.n_node*self.n_edge_type]
        A_out = self.adj_matrix[:,:, self.n_node*self.n_edge_type:]        

        with tf.variable_scope(self.scope):
            a_in = tf.matmul(A_in, self.in_states)
            a_out = tf.matmul(A_out, self.out_states)
            a = tf.concat([a_in, a_out, self.cur_states], 2)

            r = tf.layers.dense(a, self.node_state_dim, activation=tf.nn.sigmoid)
            z = tf.layers.dense(a, self.node_state_dim, activation=tf.nn.sigmoid)
            joint_h = tf.concat( [a_in, a_out, r * cur_states], 2 )
            h_hat = tf.layers.dense( joint_h, self.node_state_dim, activation=tf.nn.tanh)

            output = (1 - z) * cur_states + z * h_hat

    def __call__(self, in_states, out_states, cur_states, adj_matrix):


        return 

    def predict(self,
            in_states,
            out_states,
            cur_states,
            adj_matrix,
            reuse=False,
        ):

        with tf.variable_scope(self.scope, reuse=reuse):




if __name__ == '__main__':
    pass