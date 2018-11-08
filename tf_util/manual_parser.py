# 
import init_path

# compute import
import numpy as np
import tensorflow as tf

def manual_parser(env):

    def _half_cheetah():

        graph = np.array([
                [0, 1, 0, 0, 1, 0, 0],
                [1, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 1, 0]
            ])

        geom_meta_info = np.array([
                [0.046,     1,     0, 0,      0, 0, 0, 0,     0],
                [0.046, 0.145,   0.1, 0,  -0.13, 0, 1, 0,  -3.8],
                [0.046,  0.15, -0.14, 0,  -0.07, 0, 1, 0, -2.03],
                [0.046, 0.094,  0.03, 0, -0.097, 0, 1, 0, -0.27],
                [0.046, 0.133, -0.07, 0,  -0.12, 0, 1, 0,  0.52], 
                [0.046, 0.106, 0.065, 0,  -0.09, 0, 1, 0,  -0.6],
                [0.046,  0.07, 0.045, 0,  -0.07, 0, 1, 0,  -0.6]
            ])
        joint_meta_info = np.array([
                [  0,      0,     0,   0],
                [  6,  -0.52,  1.05, 240],
                [4.5, -0.785, 0.785, 180],
                [  3,    0.4, 0.785, 120],
                [4.5,     -1,   0.7, 180],
                [  3,   -1.2,  0.87, 120],
                [1.5,   -0.5,   0.5,  60]
            ])
        meta_info = np.hstack( (geom_meta_info, joint_meta_info) )

        ob_assign = np.array([
                0, 0, 1, 2, 3, 4, 5, 6,\
                0, 0, 0, 1, 2, 3, 4, 5, 6
            ])

        ac_assign = np.array([
                1, 2, 3, 4, 5, 6
            ])


        return graph, ob_assign, ac_assign

    if env == 'HalfCheetah-v2': return _half_cheetah()
    else: 
        raise NotImplementedError

def verify_symmetric_mat(mat):
    N, _ = mat.shape
    for i in range(N):
        for j in range(N):
            assert mat[i, j] == mat[j, i]
    return True


if __name__ == '__main__':
    from config import config
    from data import data
    args = config.get_config()

    graph = manual_parser('HalfCheetah-v2')

    
    import pdb; pdb.set_trace()
    pass