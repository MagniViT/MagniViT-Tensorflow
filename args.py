import argparse
import os
import random
import numpy as np
import tensorflow as tf
#/run/user/1000/gvfs/smb-share:server=rds.icr.ac.uk,share=data/DBI/DUDBI/DYNCESYS/OlgaF/camelyon16/
#/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/camelyon16/CLAM-features/

def parse_args():
    """Parse input arguments.
    Parameters
    -------------------
    No parameters.
    Returns
    -------------------
    args: argparser.Namespace class object
        An argparse.Namespace class object contains experimental hyper-parameters.
    """
    parser = argparse.ArgumentParser(description='Train Graph Att net')
    parser.add_argument('--strategy', dest='strategy',
                        help='strategy to pick for co-training',
                        default='Alternating', type=str,
                        choices=['Task by task','Alternating','Accumulating gradients'])
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory where the weights of the model are stored',
                        default="Saved_model", type=str)
    parser.add_argument('--lr', dest='init_lr',
                        help='initial learning rate',
                        default=0.0002, type=float)
    parser.add_argument('--decay', dest='weight_decay',
                        help='weight decay',
                        default=1e-5, type=float)
    parser.add_argument('--momentum', dest='momentum',
                        help='momentum',
                        default=0.9, type=float)
    parser.add_argument('--epoch', dest='max_epoch',
                        help='number of epochs to train GRAPH MIL',
                        default=100, type=int)
    parser.add_argument('--n_classes', dest='n_classes',
                        help='number of classes',
                        default=2, type=int)
    parser.add_argument('--seed_value', dest='seed_value',
                        help='use same seed value for reproducability',
                        default=12321, type=int)
    parser.add_argument('--run', dest='run',
                        help='number of experiments to be run',
                        default=5, type=int)
    parser.add_argument('--n_folds', dest='n_folds',
                        help='number of folds in the cross fold validation',
                        default=1, type=int)
    parser.add_argument('--feature_path', dest='feature_path',
                        help='directory where the images are stored',
                        default="/run/user/1001/gvfs/smb-share:server=rds.icr.ac.uk,share=data/DBI/DUDBI/DYNCESYS/OlgaF/camelyon_data/camelyon_features/", type=str)
    parser.add_argument('--experiment_name', dest='experiment_name',
                        help='the name of the experiment needed for the logs',
                        default="test", type=str)
    parser.add_argument('--input_shape', dest="input_shape",
                        help='shape of the image',
                        default=(1024,), type=int, nargs=3)
    parser.add_argument('--csv', dest="csv_file",
                        help='csv file with information about the labels',
                       default="csv_files/fold_0.csv",type=str)

    args = parser.parse_args()
    return args


def set_seed(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
