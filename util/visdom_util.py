''' visdom related functions
'''

from visdom import Visdom
import time
import numpy as np

viz = None
def visdom_initialize(args):
    '''
    '''
    global viz

    if args.vis_port < 4212 or args.vis_port > 4223:
        assert 0, 'Visdom port %d not supported' % (args.vis_port)

    if args.vis_server == 'local':
        viz = Visdom()
    else:
        viz = Visdom(server=args.vis_server, port=args.vis_port)

    return None

def viz_line(i, vals, viz_win=None, 
             title='', xlabel='', ylabel='', 
             legend=None):
    ''' a more robust way to print multiple values on the same plot
        NOTE:
            this function only supports 2 or more valus. for plotting only one value,
            refer to visdom_plot_curve()
        UPDATE:
            now it data input supports >= 1 plotting
    '''
    data_num = len(vals)
    if legend != None: assert data_num == len(legend)
    else: legend = ['' for _ in range(data_num)]

    # make the input compatible with visdom API
    X = [np.array([i]) for _ in range(data_num)]
    Y = [np.array([val]) for val in vals]
    if data_num != 1:
        X = np.column_stack(X)
        Y = np.column_stack(Y)
    else:
        X = X[-1]
        Y = Y[-1]
    
    if viz_win is None:
        return viz.line(X=X, Y=Y,
                        opts=dict(title=title,
                                  legend=legend,
                                  xlabel=xlabel,
                                  ylabel=ylabel))
    else:
        return viz.line(win=viz_win, update='append',
                        X=X, Y=Y,
                        opts=dict(title=title,
                                  legend=legend))