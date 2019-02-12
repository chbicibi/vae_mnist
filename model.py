import argparse
import glob
import os
import re
import shutil
from functools import reduce
from itertools import chain
from time import sleep

import numpy as np
import matplotlib.pyplot as plt

import chainer
import chainer.functions as F
import chainer.links as L

import myutils as ut

import common as C_
import net_ae as NA_
import net_vae as NV_


DEBUG0 = False


################################################################################
# モデル
################################################################################

def get_model_case0():
    ''' 畳み込みオートエンコーダ '''

    # 入出力チャンネル数を指定
    model = NA_.CAEList(
        NA_.CAEChain(2, 10, activation=(F.relu, None)), # in: 512
        NA_.CAEChain(10, 20), # in: 256
        NA_.CAEChain(20, 20), # in: 128
        NA_.CAEChain(20, 20), # in: 64
        NA_.CAEChain(20, 20), # in: 32
        NA_.CAEChain(20, 20), # in: 16
        NA_.CAEChain(20, 20), # in: 8
        NA_.CAEChain(20, 20), # in: 3
        NA_.LAEChain(None, 20), # in: 10*3*3
        NA_.LAEChain(20, 10, activation=(None, F.relu)))
    return model


def get_model_case1():
    ''' VAE '''

    # 入出力チャンネル数を指定
    model = NA_.CAEList(
        NA_.CAEChain(2, 10, activation=(F.relu, None)), # in: 512, 1024
        NA_.CAEChain(10, 20), # in: 256
        NA_.CAEChain(20, 20), # in: 128
        NA_.CAEChain(20, 20), # in: 64
        NA_.CAEChain(20, 20), # in: 32
        NA_.CAEChain(20, 20), # in: 16
        NA_.CAEChain(20, 20), # in: 8
        NA_.CAEChain(20, 20), # in: 4
        NA_.LAEChain(None, 20), # in: 10*3*3
        NV_.VAEChain(20, 10)) # in: 10*3*3

    loss = NV_.VAELoss(model, beta=1.0, k=1)
    return loss


def get_model_case1_z30():
    ''' VAE '''

    # 入出力チャンネル数を指定
    model = NA_.CAEList(
        NA_.CAEChain(2, 10, activation=(F.relu, None)), # in: 512, 1024
        NA_.CAEChain(10, 20), # in: 256
        NA_.CAEChain(20, 20), # in: 128
        NA_.CAEChain(20, 20), # in: 64
        NA_.CAEChain(20, 20), # in: 32
        NA_.CAEChain(20, 20), # in: 16
        NA_.CAEChain(20, 20), # in: 8
        NA_.CAEChain(20, 20), # in: 4
        # NA_.LAEChain(None, 40), # in: 10*3*3
        NV_.VAEChain(None, 30)) # in: 10*3*3

    loss = NV_.VAELoss(model, beta=1.0, k=1)
    return loss


def get_model_case2n():
    ''' AE '''

    # 入出力チャンネル数を指定
    model = NA_.CAEList(
        NA_.CAEChain(2, 10, activation=(F.relu, None)), # in: 512, 1024
        NA_.CAEChain(10, 20), # in: 256
        NA_.CAEChain(20, 20), # in: 128
        NA_.CAEChain(20, 20), # in: 64
        NA_.CAEChain(20, 20), # in: 32
        NA_.CAEChain(20, 20), # in: 16
        NA_.CAEChain(20, 20), # in: 8
        NA_.CAEChain(20, 20), # in: 4
        NA_.LAEChain(None, 10), # in: 10*3*3
        NA_.LAEChain(10, 2, activation=(None, F.relu)))

    loss = L.Classifier(model, lossfun=F.mean_squared_error)
    if C_.DEVICE >= 0:
        loss.to_gpu(C_.DEVICE)
    return loss


def get_model_case2():
    ''' VAE '''

    # 入出力チャンネル数を指定
    model = NA_.CAEList(
        NA_.CAEChain(2, 10, activation=(F.relu, None)), # in: 512, 1024
        NA_.CAEChain(10, 20), # in: 256
        NA_.CAEChain(20, 20), # in: 128
        NA_.CAEChain(20, 20), # in: 64
        NA_.CAEChain(20, 20), # in: 32
        NA_.CAEChain(20, 20), # in: 16
        NA_.CAEChain(20, 20), # in: 8
        NA_.CAEChain(20, 20), # in: 4
        NA_.LAEChain(None, 10), # in: 10*3*3
        NV_.VAEChain(10, 2)) # in: 10*3*3

    loss = NV_.VAELoss(model, beta=1.0, k=1)
    return loss


def get_model_case3():
    ''' VAE '''

    # 入出力チャンネル数を指定
    model = NA_.CAEList(
        NA_.CAEChain(2, 10, activation=(F.relu, None)), # in: 512, 1024
        NA_.CAEChain(10, 20), # in: 256
        NA_.CAEChain(20, 20), # in: 128
        NA_.CAEChain(20, 20), # in: 64
        NA_.CAEChain(20, 20), # in: 32
        NA_.CAEChain(20, 20), # in: 16
        NA_.CAEChain(20, 20), # in: 8
        NA_.CAEChain(20, 20), # in: 4
        NA_.LAEChain(None, 10), # in: 10*3*3
        NV_.VAEChain(10, 3)) # in: 10*3*3

    loss = NV_.VAELoss(model, beta=1.0, k=1)
    return loss


def get_model_case4_0():
    ''' VAE
    1 4 10 22 46 94 190 382
    2 6 14 30 62 126 254 510
    3 8 18 38 78 158 318
    '''

    # 入出力チャンネル数を指定
    model = NA_.CAEList(
        NA_.CAEChain(2, 10, activation=(F.relu, None)),
        NA_.CAEChain(10, 20),
        NA_.CAEChain(20, 30),
        NA_.CAEChain(30, 30),
        NA_.CAEChain(30, 30),
        NA_.CAEChain(30, 30),
        NA_.CAEChain(30, 20),
        NV_.VAEChain(None, 10))

    loss = NV_.VAELoss(model, beta=1.0, k=1)
    return loss


def get_model_case4_1():
    ''' VAE
    1 4 10 22 46 94 190 382
    2 6 14 30 62 126 254 510
    3 8 18 38 78 158 318
    '''

    # 入出力チャンネル数を指定
    model = NA_.CAEList(
        NA_.CAEChain(2, 10, activation=(F.relu, None)),
        NA_.CAEChain(10, 20),
        NA_.CAEChain(20, 30),
        NA_.CAEChain(30, 30),
        NA_.CAEChain(30, 30),
        NA_.CAEChain(30, 30),
        NA_.CAEChain(30, 20),
        NV_.VAEChain(None, 10))

    loss = NV_.VAELoss(model, beta=1.0, k=1)
    return loss


def get_model_case4_2():
    ''' VAE
    1 4 10 22 46 94 190 382
    2 6 14 30 62 126 254 510
    3 8 18 38 78 158 318
    '''

    # 入出力チャンネル数を指定
    model = NA_.CAEList(
        NA_.CAEChain(2, 8, activation=(F.relu, None)),
        NA_.CAEChain(8, 16),
        NA_.CAEChain(16, 32),
        NA_.CAEChain(32, 64),
        NA_.CAEChain(64, 64),
        NA_.CAEChain(64, 128),
        NA_.CAEChain(128, 256),
        NV_.VAEChain(None, 64))

    loss = NV_.VAELoss(model, beta=1.0, k=1)
    return loss


def get_model_case5_0():
    ''' AE
    1 4 10 22 46 94 190 382
    2 6 14 30 62 126 254 510
    3 8 18 38 78 158 318
    '''

    # 入出力チャンネル数を指定
    model = NA_.CAEList(
        NA_.CAEChain(2, 10, activation=(F.relu, None)),
        NA_.CAEChain(10, 20),
        NA_.CAEChain(20, 30),
        NA_.CAEChain(30, 30),
        NA_.CAEChain(30, 30),
        NA_.CAEChain(30, 30),
        NA_.CAEChain(30, 20),
        NA_.LAEChain(None, 10, activation=(None, F.relu)))

    loss = L.Classifier(model, lossfun=F.mean_squared_error)
    if C_.DEVICE >= 0:
        loss.to_gpu(C_.DEVICE)
    return loss


def get_model_case5_1():
    ''' AE
    1 4 10 22 46 94 190 382
    2 6 14 30 62 126 254 510
    3 8 18 38 78 158 318
    '''

    # 入出力チャンネル数を指定
    model = NA_.CAEList(
        NA_.CAEChain(2, 10, activation=(F.relu, None)),
        NA_.CAEChain(10, 20),
        NA_.CAEChain(20, 30),
        NA_.CAEChain(30, 30),
        NA_.CAEChain(30, 30),
        NA_.CAEChain(30, 30),
        NA_.CAEChain(30, 20),
        NA_.LAEChain(None, 10, activation=(None, F.relu)))

    loss = L.Classifier(model, lossfun=F.mean_squared_error)
    if C_.DEVICE >= 0:
        loss.to_gpu(C_.DEVICE)
    return loss


def get_model_case6():
    ''' VAE
    1 4 10 22 46 94 190 382
    2 6 14 30 62 126 254 510
    3 8 18 38 78 158 318
    '''

    # 入出力チャンネル数を指定
    model = NA_.CAEList(
        NA_.CAEChain(3, 8, activation=(F.relu, None)),
        NA_.CAEChain(8, 16),
        NA_.CAEChain(16, 32),
        NA_.CAEChain(32, 64),
        NA_.CAEChain(64, 64),
        NA_.CAEChain(64, 128),
        NA_.CAEChain(128, 256),
        NV_.VAEChain(None, 64))

    loss = NV_.VAELoss(model, beta=1.0, k=1)
    return loss


def get_model_case6_1():
    ''' VAE
    '''

    # 入出力チャンネル数を指定
    model = NA_.CAEList(
        NA_.CAEChain(3, 8, activation=(F.relu, None), batch_norm=True),
        NA_.CAEChain(8, 16, batch_norm=True),
        NA_.CAEChain(16, 32, batch_norm=True),
        NA_.CAEChain(32, 64, batch_norm=True),
        NA_.CAEChain(64, 64, batch_norm=True),
        NA_.CAEChain(64, 128, batch_norm=True),
        NA_.CAEChain(128, 256, batch_norm=True),
        NV_.VAEChain(None, 64))

    loss = NV_.VAELoss(model, beta=1.0, k=1)
    return loss


def get_model_case6_2():
    ''' VAE
    '''

    # 入出力チャンネル数を指定
    model = NA_.CAEList(
        NA_.CAEChain(3, 8, activation=(F.relu, None), batch_norm='re'),
        NA_.CAEChain(8, 16, batch_norm='re'),
        NA_.CAEChain(16, 32, batch_norm='re'),
        NA_.CAEChain(32, 64, batch_norm='re'),
        NA_.CAEChain(64, 64, batch_norm='re'),
        NA_.CAEChain(64, 128, batch_norm='re'),
        NA_.CAEChain(128, 256, batch_norm='re'),
        NV_.VAEChain(None, 64))

    loss = NV_.VAELoss(model, beta=1.0, k=1)
    return loss


def get_model_case6_3():
    ''' VAE
    '''

    # 入出力チャンネル数を指定
    model = NA_.CAEList(
        NA_.CAEChain(3, 8, activation=(F.relu, None), batch_norm='re',
                    padding=True),
        NA_.CAEChain(8, 16, batch_norm='re', padding=True),
        NA_.CAEChain(16, 32, batch_norm='re', padding=True),
        NA_.CAEChain(32, 64, batch_norm='re', padding=True),
        NA_.CAEChain(64, 64, batch_norm='re', padding=True),
        NA_.CAEChain(64, 128, batch_norm='re', padding=True),
        NA_.CAEChain(128, 256, batch_norm='re', padding=True),
        NV_.VAEChain(None, 64))

    loss = NV_.VAELoss(model, beta=1.0, k=1)
    return loss


def get_model_case6_4():
    ''' VAE
    '''

    # 入出力チャンネル数を指定
    model = NA_.CAEList(
        NA_.CAEChain(3, 8, activation=(F.relu, None)),
        NA_.CAEChain(8, 16, activation=F.relu),
        NA_.CAEChain(16, 32, activation=F.relu),
        NA_.CAEChain(32, 64, activation=F.relu),
        NA_.CAEChain(64, 64, activation=F.relu),
        NA_.CAEChain(64, 128, activation=F.relu),
        NA_.CAEChain(128, 256, activation=F.relu),
        NV_.VAEChain(None, 64, activation=F.relu))

    loss = NV_.VAELoss(model, beta=1.0, k=1)
    return loss


def get_model_case7():
    ''' AE
    '''

    # 入出力チャンネル数を指定
    model = NA_.CAEList(
        NA_.CAEChain(3, 8, activation=(F.relu, None), batch_norm='re',
                    ksize=5, padding=True),
        NA_.CAEChain(8, 16, batch_norm='re', ksize=5, padding=True),
        NA_.CAEChain(16, 32, batch_norm='re', ksize=5, padding=True),
        NA_.CAEChain(32, 64, batch_norm='re', ksize=5, padding=True),
        NA_.CAEChain(64, 64, batch_norm='re', ksize=5, padding=True),
        NA_.CAEChain(64, 128, batch_norm='re', ksize=5, padding=True),
        NA_.CAEChain(128, 256, batch_norm='re', ksize=5, padding=True),
        NA_.LAEChain(None, 64, activation=(None, F.relu)))

    loss = L.Classifier(model, lossfun=F.mean_squared_error)
    if C_.DEVICE >= 0:
        loss.to_gpu(C_.DEVICE)
    return loss


def get_model_case7_1():
    ''' AE
    '''

    # 入出力チャンネル数を指定
    model = NA_.CAEList(
        NA_.CAEChain(3, 8, activation=(F.sigmoid, None)),
        NA_.CAEChain(8, 16, activation=F.sigmoid),
        NA_.CAEChain(16, 32, activation=F.sigmoid),
        NA_.CAEChain(32, 64, activation=F.sigmoid),
        NA_.CAEChain(64, 64, activation=F.sigmoid),
        NA_.CAEChain(64, 128, activation=F.sigmoid),
        NA_.CAEChain(128, 256, activation=F.sigmoid),
        NA_.LAEChain(None, 64, activation=(None, F.sigmoid)))

    loss = NA_.AELoss(model)
    return loss


def get_model_case8_0():
    ''' VAE
    activation: F.relu => F.sigmoid
    '''

    # 入出力チャンネル数を指定
    model = NA_.CAEList(
        NA_.CAEChain(3, 8, activation=(F.sigmoid, None)),
        NA_.CAEChain(8, 16, activation=F.sigmoid),
        NA_.CAEChain(16, 32, activation=F.sigmoid),
        NA_.CAEChain(32, 64, activation=F.sigmoid),
        NA_.CAEChain(64, 64, activation=F.sigmoid),
        NA_.CAEChain(64, 128, activation=F.sigmoid),
        NA_.CAEChain(128, 256, activation=F.sigmoid), # => (256, 3, 3)
        NV_.VAEChain(None, 64, activation=(None, F.sigmoid))) # 2304 -> 64 (1 / 36)

    loss = NV_.VAELoss(model, beta=1.0, k=1)
    return loss


def get_model_case9_0():
    ''' VAE
    z: 2dim
    '''
    model = NA_.CAEList(
        NA_.CAEChain(3, 8, activation=(F.sigmoid, None)),
        NA_.CAEChain(None, 16),
        NA_.CAEChain(None, 32),
        NA_.CAEChain(None, 64),
        NA_.CAEChain(None, 64),
        NA_.CAEChain(None, 64),
        NA_.CAEChain(None, 128), # => (128, 3, 3)
        NA_.LAEChain(None, 128), # 1152 -> 128 (1 / 9)
        NA_.LAEChain(None, 16), # 128 -> 16 (1 / 8)
        NV_.VAEChain(None, 2)) # 16 -> 2 (1 / 8)

    loss = NV_.VAELoss(model)
    return loss


def get_model_case9_1():
    ''' VAE
    z: 32dim
    '''
    model = NA_.CAEList(
        NA_.CAEChain(3, 8, activation=(F.sigmoid, None)),
        NA_.CAEChain(None, 16),
        NA_.CAEChain(None, 32),
        NA_.CAEChain(None, 64),
        NA_.CAEChain(None, 64),
        NA_.CAEChain(None, 64),
        NA_.CAEChain(None, 128), # => (128, 3, 3)
        NA_.LAEChain(None, 288), # 1152 -> 288 (1 / 4)
        NA_.LAEChain(None, 96), # 288 -> 96 (1 / 3)
        NV_.VAEChain(None, 32)) # 96 -> 32 (1 / 3)

    loss = NV_.VAELoss(model)
    return loss


def get_model_case9_2():
    ''' VAE
    z: 64dim
    パラメータ数: ((3*8+8*8+8*16+16*16+16*32+32*32+32*64+64*64+64*64+64*64*128+128*128+128*128+128*128)*5*5+
                1152*384+384*128+128*64)*2=30283824
    '''
    model = NA_.CAEList(
        NA_.CAEChainM(3, 8, activation=(F.sigmoid, None), ksize=5, n_conv=2),
        NA_.CAEChainM(None, 16, ksize=5, n_conv=2),
        NA_.CAEChainM(None, 32, ksize=5, n_conv=2),
        NA_.CAEChainM(None, 64, ksize=5, n_conv=2),
        NA_.CAEChainM(None, 64, ksize=5, n_conv=2),
        NA_.CAEChainM(None, 128, ksize=5, n_conv=2),
        NA_.CAEChainM(None, 128, ksize=5, n_conv=2), # => (128, 3, 3)
        NA_.LAEChain(None, 384), # 1152 -> 384 (1 / 3)
        NA_.LAEChain(None, 128), # 384 -> 128/ (1 / 3)
        NV_.VAEChain(None, 64)) # 128 -> 64 (1 / 2)

    loss = NV_.VAELoss(model)
    return loss


def get_model_case9_3():
    ''' VAE
    z: 64dim
    パラメータ数: ((3*8+8*8+8*16+16*16+16*32+32*32+32*64+64*64+64*64+64*64*128+128*128+128*128+128*128)*3*3+
                1152*384+384*128+128*64)*2=11541808
    '''
    model = NA_.CAEList(
        NA_.CAEChainM(3, 8, activation=(F.sigmoid, None), ksize=3, n_conv=2),
        NA_.CAEChainM(None, 16, ksize=3, n_conv=2),
        NA_.CAEChainM(None, 32, ksize=3, n_conv=2),
        NA_.CAEChainM(None, 64, ksize=3, n_conv=2),
        NA_.CAEChainM(None, 64, ksize=3, n_conv=2),
        NA_.CAEChainM(None, 128, ksize=3, n_conv=2),
        NA_.CAEChainM(None, 128, ksize=3, n_conv=2), # => (128, 3, 3)
        NA_.LAEChain(None, 384), # 1152 -> 384 (1 / 3)
        NA_.LAEChain(None, 128), # 384 -> 128 (1 / 3)
        NV_.VAEChain(None, 64)) # 128 -> 64 (1 / 2)

    loss = NV_.VAELoss(model)
    return loss


def get_model_case9_4():
    ''' VAE
    z: 64dim
    パラメータ数: ((3*8+8*8+8*16+16*16+16*32+32*32+32*64+64*64+64*64+64*64*128+128*128+128*128+128*128)*3*3+
                1152*384+384*128+128*64)*2=11541808
    '''
    model = NA_.CAEList(
        NA_.CAEChain(3, 8, activation=(F.sigmoid, None), ksize=3),
        NA_.CAEChain(None, 16, ksize=3),
        NA_.CAEChain(None, 32, ksize=3),
        NA_.CAEChain(None, 64, ksize=3),
        NA_.CAEChain(None, 64, ksize=3),
        NA_.CAEChain(None, 128, ksize=3),
        NA_.CAEChain(None, 128, ksize=3), # => (128, 3, 3)
        NA_.LAEChain(None, 384), # 1152 -> 384 (1 / 3)
        NA_.LAEChain(None, 128), # 384 -> 128 (1 / 3)
        NV_.VAEChain(None, 64)) # 128 -> 64 (1 / 2)

    loss = NV_.VAELoss(model)
    return loss


################################################################################

def get_model_case10_0(model_type='vae'):
    if model_type == 'vae':
        last_layer = NV_.VAEChain
        loss_chain = NV_.VAELoss
    else:
        last_layer = NA_.LAEChain
        loss_chain = NA_.AELoss

    model = NA_.CAEList(
        NA_.CAEChain(None, 16, activation=(F.sigmoid, None)), # => 14
        NA_.CAEChain(None, 32), # => 7
        NA_.CAEChain(None, 64), # => 4
        NA_.CAEChain(None, 128), # => 2 => 512
        NA_.LAEChain(None, 64),
        NA_.LAEChain(None, 16),
        last_layer(None, 2, activation=(None, F.sigmoid)))

    return loss_chain(model)


def get_model_case10_1(model_type='vae'):
    if model_type == 'vae':
        last_layer = NV_.VAEChain
        loss_chain = NV_.VAELoss
    else:
        last_layer = NA_.LAEChain
        loss_chain = NA_.AELoss

    model = NA_.CAEList(
        NA_.CAEChain(None, 16, activation=(F.sigmoid, None)), # => 14
        NA_.CAEChain(None, 32), # => 7
        NA_.CAEChain(None, 64), # => 4
        NA_.CAEChain(None, 128), # => 2 => 512
        NA_.LAEChain(None, 256),
        NA_.LAEChain(None, 128),
        last_layer(None, 64))

    return loss_chain(model)


def get_model_case10_2(model_type='vae'):
    if model_type == 'vae':
        last_layer = NV_.VAEChain
        loss_chain = NV_.VAELoss
    else:
        last_layer = NA_.LAEChain
        loss_chain = NA_.AELoss

    model = NA_.CAEList(
        NA_.CAEChain(None, 16, activation=(F.sigmoid, None)), # => 14
        NA_.CAEChain(None, 32), # => 7
        NA_.CAEChain(None, 64), # => 4
        NA_.CAEChain(None, 128), # => 2 => 512
        NA_.LAEChain(None, 128),
        NA_.LAEChain(None, 32),
        last_layer(None, 3))

    return loss_chain(model)


def get_model_case10_fc_z2(model_type='vae'):
    if model_type == 'vae':
        last_layer = NV_.VAEChain
        loss_chain = NV_.VAELoss
    else:
        last_layer = NA_.LAEChain
        loss_chain = NA_.AELoss

    model = NA_.CAEList(
        NA_.LAEChain(None, 196, activation=(F.sigmoid, None)),
        NA_.LAEChain(None, 49),
        NA_.LAEChain(None, 7),
        last_layer(None, 2))

    return loss_chain(model)


def get_model(name, sample=None):
    # 関数名自動取得に変更

    if 'vae' in name:
        model_type = 'vae'
        name = re.sub(r'_?vae', '', name)
        print(name)
    elif 'ae' in name:
        model_type = 'ae'
        name = re.sub(r'_?ae', '', name)
    else:
        model_type = 'vae'

    function_name = 'get_model_' + name
    if function_name in globals():
        model = globals()[function_name](model_type=model_type)
    else:
        raise NameError('Function Not Found:', function_name)

    if sample is not None:
        # モデル初期化
        print('init model')
        if sample.ndim == 3:
            sample = sample[None, ...]
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            model.predictor(model.xp.asarray(sample), show_shape=True)
    return model
