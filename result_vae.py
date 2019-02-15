import argparse
import glob
import itertools
import os
import re
import shutil
import sys
import traceback
from itertools import chain, islice
from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plc

import chainer
import chainer.functions as F
import chainer.iterators as I
import chainer.optimizers as O
import chainer.training as T
import chainer.training.extensions as E

import myutils as ut

import common as C_
import dataset as D_
import model as M_
import vis as V_
from train_vae import check_snapshot


# パス
SRC_DIR = os.path.dirname(__file__)
SRC_FILE = os.path.basename(__file__)
SRC_FILENAME = os.path.splitext(SRC_FILE)[0]


################################################################################

def get_shaped_mnist():
    def f_(it):
        return [(x[0].reshape(1, 28, 28), x[1]) for x in it]

    train, test = map(f_, chainer.datasets.get_mnist())
    return train, test


def get_data(casename):
    train, test = (list(map(itemgetter(0), d)) for d in get_shaped_mnist())
    sample = train[0]
    model = M_.get_model(casename, sample=sample)

    return model, train, test


def asarray(data):
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, C_.xp.core.core.ndarray):
        return chainer.cuda.to_cpu(data)
    elif isinstance(data, chainer.Variable):
        return asarray(data.array)
    else:
        raise TypeError(type(data))


def plot0(casename, out):
    init_file = check_snapshot(out)
    if casename == 'auto':
        casename = re.search(r'case.+?(?=[# ])', init_file)[0]

    model, train, valid = get_data(casename)
    chainer.serializers.load_npz(init_file, model, path='updater/model:main/')
    data_mnist = get_shaped_mnist()[1]

    label_no = 9
    with V_.FigDriver(1, 2) as fd:
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            for X, label in data_mnist:
                if label_no is not None and not label == label_no:
                    continue
                fd.cla()
                Z = asarray(model.encode(X[None, ...], inference=True)[0])
                # Y = asarray(model.predict(X[None, ...])[0])
                Y = asarray(model.decode(Z[None, ...], inference=True)[0])
                print(Z)
                fd[0].imshow(X[0])
                fd[1].imshow(Y[0])
                plt.pause(0.1)


def task0(*args, **kwargs):
    casename = kwargs.get('case') or 'auto'

    try:
        out = '__result__'
        plot0(casename, out)

    except Exception as e:
        error = e
        tb = traceback.format_exc()
        print('Error:', error)
        print(tb)


################################################################################

def task1(*args, **kwargs):
    data_mnist = get_shaped_mnist()[0]
    init_file = check_snapshot('__result__')

    casename = kwargs.get('case') or re.search(r'case.+?(?=[# ])', init_file)[0]
    model = M_.get_model(casename, sample=data_mnist[0][0])
    chainer.serializers.load_npz(init_file, model, path='updater/model:main/')

    def f_(label_no=None):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            for data, label in data_mnist:
                if label_no is not None and not label == label_no:
                    continue
                X = data[None, ...]
                Z = asarray(model.encode(X, inference=True)[0])
                yield Z

    # a = list(islice(f_(), 10))
    with V_.FigDriver(1, 1) as fd:
        for i in range(10):
            data = np.array(list(islice(f_(i), 100)))[:, :2]
            fd[0].scatter(*zip(data.T), label=f'{i}')
        fd.fig.legend()
        plt.show()


################################################################################

def __test__():
    pass


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', default='0',
                        choices=['', '0', '1'],
                        help='Number of main procedure')
    parser.add_argument('--case', '-c', default='',
                        help='Training case name')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--no-progress', '-p', action='store_true',
                        help='Hide progress bar')

    # additional options
    parser.add_argument('--check-snapshot', '-s', action='store_true',
                        help='Print names in snapshot file')
    parser.add_argument('--resume', '-r', nargs='?', const='all', default=None,
                        choices=['', 'model', 'all', 'modelnew', 'allnew'],
                        help='Resume with loading snapshot')

    # test option
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run as test mode')
    args = parser.parse_args()
    return args


def main():
    global DEVICE, PROGRESSBAR

    args = get_args()

    if args.gpu:
        C_.DEVICE = args.gpu

    C_.SHOW_PROGRESSBAR = not args.no_progress

    # out = args.out
    out = f'result/{SRC_FILENAME}'

    if args.test:
        # print(vars(args))
        __test__()

    elif args.mode:
        taskname = 'task' + args.mode
        f_ = globals().get(taskname)
        if f_:
            with ut.stopwatch(taskname) as sw:
                f_(**vars(args), sw=sw)


if __name__ == '__main__':
    sys.exit(main())
