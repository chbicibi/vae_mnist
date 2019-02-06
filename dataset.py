import argparse
import configparser
import math
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt

import myutils as ut

import common as C_


DEBUG0 = False


################################################################################
# 学習イテレータ
################################################################################

class ContainerBase(object):

    def __init__(self, it):
        self.it = it
        self.data = None
        self.len = len(it)

    def __len__(self):
        return self.len

    def __getitem__(self, key):
        if type(key) is tuple:
            head, *tail = key
            if type(head) is slice:
                tail = (slice(None), *tail)
            return np.array(self[head])[tail]

        elif type(key) is slice:
            # return [self[i] for i in range(*key.indices(self.len))]
            return SlicedContainer(self, *key.indices(self.len))

        else:
            if key >= self.len:
                raise IndexError
            if not self.data:
                return self.get_data(key)
            if self.data[key] is None:
                self.data[key] = self.get_data(key)
            return self.data[key]

        # else:
        #     raise TypeError

    def get_data(self, key):
        return self.it[key]


class SlicedContainer(ContainerBase):

    def __init__(self, it, start=None, stop=None, step=None):
        super().__init__(it)
        self.start = start or 0
        self.stop = stop or self.len
        self.step = step or 1
        self.last = self.stop - (self.stop - self.start) % self.step
        self.len = math.ceil((self.stop - self.start) / self.step)

    def get_data(self, key):
        if 0 <= key < self.len:
            k = self.start + key * self.step
        elif -self.len <= key < 0:
            k = self.last + (1 + key) * self.step
        else:
            raise IndexError
        return self.it[k]


class MemoizeMapList(ContainerBase):
    ''' 入力イテラブルを加工するイテラブルオブジェクト '''

    def __init__(self, fn, it, name='', cache=False, cache_path=None):
        self.name = name
        self.fn = fn
        self.it = it
        self.len = len(it)

        if cache:
            self.data = [None] * self.len
        else:
            self.data = None

        if cache_path:
            abspath = os.path.abspath(cache_path)
            os.makedirs(abspath, exist_ok=True)
            self.cache_path = abspath
        else:
            self.cache_path = None

    def get_data(self, key):
        if self.cache_path:
            if self.name:
                file = f'cache_{self.name}_{key}.npy'
            else:
                file = f'cache_{key}.npy'
            path = os.path.join(self.cache_path, file)

            if os.path.isfile(path):
                if DEBUG0:
                    print(f'load(cache) {key}/{self.len}', ' '*20, end='\r')
                return np.load(path)
            else:
                data = self.load_data(key)
                np.save(path, data)
                return data

        else:
            return self.load_data(key)

    def load_data(self, key):
        if self.fn:
            return self.fn(self.it[key])
        else:
            return self.it[key]


class MapChain(ContainerBase):
    ''' 入力イテラブルを加工するイテラブルオブジェクト
    複数のイテラブルを連結
    '''

    def __init__(self, fn, *its, name=''):
        self.name = name
        self.fn = fn
        self.its = its
        self.lens = list(map(len, its))
        self.len = sum(self.lens)
        self.data = None

    def get_data(self, key):
        if self.fn:
            return self.fn(self.point(key))
        else:
            return self.point(key)

    def point(self, key):
        if key < 0:
            key += self.len
        for i, n in enumerate(self.lens):
            if key < n:
                return self.its[i][key]
            key -= n
        print(key, self.lens)
        raise IndexError



################################################################################
# データを加工(オリジナル→) # frame => (H, W, C=[u, v, p, f, w])
################################################################################

class  Formatter(object):

    def __init__(self, vmin, vmax):
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, frame):
        a = frame[:, :, :2]
        a = (a - self.vmin) / (self.vmax - self.vmin)
        return a.transpose(2, 0, 1) # => (H, W, C) -> (C, H, W)


################################################################################

def __test__():
    pass


def get_args():
    '''
    docstring for get_args.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('out', nargs='?', default='new_script',
                        help='Filename of the new script')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run as test mode')
    args = parser.parse_args()
    return args


def main():
    '''
    docstring for main.
    '''
    args = get_args()

    if args.test:
        __test__()
        return


if __name__ == '__main__':
    main()
