import argparse
import glob
import itertools
import os
import shutil
import sys
import traceback
from itertools import chain
from operator import itemgetter
from time import sleep

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


# パス
SRC_DIR = os.path.dirname(__file__)
SRC_FILE = os.path.basename(__file__)
SRC_FILENAME = os.path.splitext(SRC_FILE)[0]

LOCK_FILE = os.path.join(SRC_DIR, f'{ut.snow}.lock')

################################################################################
# 学習
################################################################################

def plot_loss_ex(trainer):
    fig, ax = plt.subplots()
    for d in ['top', 'right']:
        ax.spines[d].set_visible(False)
    ylim_low = float('inf')
    ylim_upp = 0

    try:
        with ut.chdir(trainer.out):
            if not os.path.isfile('log.json'):
                return

            log = ut.load('log.json', from_json=True)

            for key in ('main/loss', 'val/main/loss'):
                a = np.array([l[key] for l in log])
                a = np.clip(a, 0, 1e6)

                ax.plot(a, label=key)
                ylim_upp = max(np.ceil(np.max(a[min(len(a)-1,3):50]))/1000*1000,
                               ylim_upp)
                ylim_low = min(np.min(a)//1000*1000, ylim_low)

            ax.set_ylim((ylim_low, ylim_upp))
            ax.set_xlabel('epoch')
            ax.grid(True)
            fig.legend()
            fig.savefig('loss1.png')

    finally:
        plt.close(fig)
        sleep(10)


def lr_drop_ex(alpha, start=200):
    def f_(trainer):
        epoch = trainer.updater.epoch
        if epoch < start:
            return
        # trainer.updater.get_optimizer('main').alpha *= 0.1
        # alpha_new = alpha * max(0.8**max((epoch-start)//50+1, 0), 0.1)
        alpha_new = alpha * 0.1
        trainer.updater.get_optimizer('main').alpha = alpha_new
    return f_


def pause_ex(trainer):
    while os.path.isfile(os.path.join(trainer.out, 'pause')):
        print('[pause]', end=' \r')
        sleep(10)


def train_model(model, train_iter, valid_iter, epoch=10, out='__result__',
                init_file=None, fix_trained=False, alpha=0.001, init_all=True):
    learner = model

    # 最適化手法の選択
    optimizer = O.Adam(alpha=alpha).setup(learner)

    if fix_trained:
        for m in model[:-1]:
            m.disable_update()

    # Updaterの準備 (パラメータを更新)
    updater = T.StandardUpdater(train_iter, optimizer, device=C_.DEVICE)

    # Trainerの準備
    trainer = T.Trainer(updater, stop_trigger=(epoch, 'epoch'), out=out)

    # TrainerにExtensionを追加する
    ## 検証
    trainer.extend(E.Evaluator(valid_iter, learner, device=C_.DEVICE),
                   name='val')

    ## モデルパラメータの統計を記録する
    trainer.extend(E.ParameterStatistics(learner.predictor,
                                                  {'std': np.std},
                                                  prefix='links'))

    ## 学習率を記録する
    trainer.extend(E.observe_lr())

    ## 学習経過を画面出力
    trainer.extend(
        E.PrintReport(
            ['epoch', 'main/loss', 'val/main/loss', 'elapsed_time', 'lr']))

    ## ログ記録 (他のextensionsの結果も含まれる)
    trainer.extend(E.LogReport(log_name='log.json'))

    ## 学習経過を画像出力
    if C_.OS_IS_WIN:
        def ex_pname(link):
            ls = list(link.links())[1:]
            if not ls:
                names = (p.name for p in link.params())
            else:
                names = chain(*map(ex_pname, ls))
            return [f'{link.name}/{n}' for n in names]

        def register(keys, file_name):
            trainer.extend(E.PlotReport(keys,# x_key='epoch',
                                                 file_name=file_name,
                                                 marker=None))

        register('lr', file_name='lr.png')
        register(['main/loss', 'val/main/loss'], file_name='loss.png')

        if 'vae' in learner.name:
            register(['main/reconstr', 'val/main/reconstr'],
                     file_name='reconstr.png')

            register(['main/kl_penalty', 'val/main/kl_penalty'],
                     file_name='kl_penalty.png')

            register(['main/mse_vel', 'val/main/mse_vel'],
                     file_name='mse_vel.png')

            register(['main/mse_vor', 'val/main/mse_vor'],
                     file_name='mse_vor.png')

        for link in learner.predictor:
            param_names = ex_pname(link)
            for d in ('data', 'grad'):
                observe_keys_std = [f'links/predictor/{key}/{d}/std'
                                    for key in param_names]
                for l in ('enc', 'dec', 'bne', 'bnd'):
                    file_name = f'std_{d}_{l}_{link.name}.png'
                    f_ = lambda s: l in s# or f'bn{l[0]}' in s
                    keys = list(filter(f_, observe_keys_std))
                    register(keys, file_name=file_name)

    ## ネットワーク形状をdot言語で出力
    ## 可視化コード: ```dot -Tpng cg.dot -o [出力ファイル]```
    trainer.extend(E.dump_graph('main/loss'))

    ## トレーナーオブジェクトをシリアライズし、出力ディレクトリに保存
    trainer.extend(
        E.snapshot(filename='snapshot_epoch-{.updater.epoch}.model'))

    ## プログレスバー
    if C_.SHOW_PROGRESSBAR:
        trainer.extend(E.ProgressBar())

    if init_file:
        print('loading snapshot:', init_file)
        try:
            if init_all:
                chainer.serializers.load_npz(init_file, trainer)

            else:
                chainer.serializers.load_npz(init_file, learner,
                                             path='updater/model:main/')
        except KeyError:
            raise

    # 自作Extension
    # trainer.extend(plot_loss_ex, trigger=(1, 'epoch'))
    # trainer.extend(lr_drop_ex(alpha), trigger=(1, 'epoch'))
    trainer.extend(pause_ex, trigger=(1, 'iteration'))

    # 学習を開始する
    try:
        trainer.run()
    except:
        print('trainer except')
        raise
    finally:
        print('trainer end')


################################################################################

def check_snapshot(out, kw='', show=False):
    # モデルのパスを取得
    respath = ut.select_file(out, idx=None)
    print('path:', respath)
    file = ut.select_file(respath, key=r'snapshot_.*', idx=None)
    print('file:', file)

    if show:
        # npz保存名確認
        with np.load(file) as npzfile:
            for f in npzfile:
                print(f)
                continue
                if f[-1] == 'W':
                    print(f)
                    print(npzfile[f].shape)
    return file


def get_task_data(casename, batchsize):
    def f_(it):
        return [x[0].reshape(1, 28, 28) for x in it]

    train, test = map(f_, chainer.datasets.get_mnist())
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                                 repeat=False, shuffle=False)
    sample = train[0][None, ...]
    model = M_.get_model(casename, sample=sample)

    return model, train_iter, test_iter


def get_method(method, *args, **kwargs):
    return lambda obj: getattr(method, obj)(*args, **kwargs)


def process0(casename):
    ''' オートエンコーダ学習 '''

    # 学習パラメータ定義
    epoch = 500
    batchsize = 500
    logdir = f'__result__/{casename}#{ut.snow}'
    model, train_iter, valid_iter = get_task_data(casename, batchsize)

    while any(map(get_method('endswith', '.lock'), os.listdir(SRC_DIR))):
        sleep(10)

    try:
        with open(LOCK_FILE, 'w'):
            pass
        train_model(model, train_iter, valid_iter, epoch=epoch, out=logdir,
                    alpha=0.01)
    finally:
        os.remove(LOCK_FILE)


def process0_resume(casename, out, init_all=True, new_out=False):
    ''' オートエンコーダ学習 '''

    # 学習パラメータ定義
    epoch = 500
    batchsize = 500
    init_file = check_snapshot(out)
    if new_out:
        logdir = f'__result__/{casename}#{ut.snow}'
    else:
        logdir = os.path.dirname(init_file)
    model, train_iter, valid_iter = get_task_data(casename, batchsize)
    train_model(model, train_iter, valid_iter, epoch=epoch, out=logdir,
                init_file=init_file, alpha=0.01, init_all=init_all)


def task0(*args, **kwargs):
    ''' task0: 学習メイン '''

    error = None
    casename = kwargs.get('case', '')

    try:
        resume = kwargs.get('resume', '')
        if resume:
            init_all = not resume.startswith('m')
            new_out = 'new' in resume
            process0_resume(casename, out='__result__', init_all=init_all,
                            new_out=new_out)

        else:
            process0(casename)

    except Exception as e:
        error = e
        tb = traceback.format_exc()
        print('Error:', error)
        print(tb)

    if kwargs.get('sw', 0) < 3600:
        return

    with ut.EmailIO(None, 'ae_chainer: Task is Complete') as e:
        print(sys._getframe().f_code.co_name, file=e)
        print(ut.strnow(), file=e)
        if 'sw' in kwargs:
            print('Elapsed:', kwargs['sw'], file=e)
        if error:
            print('Error:', error)
            print(tb)


################################################################################

def __test__():
    pass


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', default='0',
                        choices=['', '0'],
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

    elif args.mode in '0123456789':
        taskname = 'task' + args.mode
        if taskname in globals():
            f_ = globals().get(taskname)
            with ut.stopwatch(taskname) as sw:
                f_(**vars(args), sw=sw)


if __name__ == '__main__':
    sys.exit(main())
