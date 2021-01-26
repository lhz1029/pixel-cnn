"""
Utilities for downloading and unpacking the CIFAR-10 dataset, originally published
by Krizhevsky et al. and hosted here: https://www.cs.toronto.edu/~kriz/cifar.html
"""

import os
import sys
import tarfile
from six.moves import urllib
import numpy as np
import scipy.io

def unpickle(file):
    fo = open(file, 'rb')
    if (sys.version_info >= (3, 0)):
        import pickle
        d = pickle.load(fo, encoding='latin1')
    else:
        import cPickle
        d = cPickle.load(fo)
    fo.close()
    return {'x': d['data'].reshape((10000,3,32,32)), 'y': np.array(d['labels']).astype(np.uint8)}

def load(data_dir, subset='train'):
    if subset=='train':
        path = os.path.join(data_dir, 'train_32x32.mat')
    elif subset=='test':
        path = os.path.join(data_dir, 'test_32x32.mat')
    else:
        raise NotImplementedError('subset should be either train or test')
    mat = scipy.io.loadmat(path)
    return mat['X'], mat['y']

class DataLoader(object):
    """ an object that generates batches of SVHN data for training """

    def __init__(self, data_dir, subset, batch_size, rng=None, shuffle=False, return_labels=False, small_test=False):
        """ 
        - data_dir is location where to store files
        - subset is train|test 
        - batch_size is int, of #examples to load at once
        - rng is np.random.RandomState object for reproducibility
        """

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_labels = return_labels

        # load CIFAR-10 training data to RAM
        self.data, self.labels = load(data_dir, subset=subset)
        self.data = np.transpose(self.data, (3,0,1,2)) # (32,32,3,N) -> (N,32,32,3)
        if small_test:
            self.data = self.data[:50]
            self.labels = self.labels[:50]
        
        self.p = 0 # pointer to where we are in iteration
        self.rng = np.random.RandomState(1) if rng is None else rng

    def get_observation_size(self):
        return self.data.shape[1:]

    def get_num_labels(self):
        return np.amax(self.labels) + 1

    def reset(self):
        self.p = 0

    def __iter__(self):
        return self

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        if n is None: n = self.batch_size

        # on first iteration lazily permute all data
        if self.p == 0 and self.shuffle:
            inds = self.rng.permutation(self.data.shape[0])
            self.data = self.data[inds]
            self.labels = self.labels[inds]

        # on last iteration reset the counter and raise StopIteration
        if self.p + n > self.data.shape[0]:
            self.reset() # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        x = self.data[self.p : self.p + n]
        y = self.labels[self.p : self.p + n]
        self.p += self.batch_size

        if self.return_labels:
            return x,y
        else:
            return x

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)


