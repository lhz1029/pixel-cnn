# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Evaluating Likelihood Ratios based on pixel_cnn model.


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import glob
import math
import argparse
import cv2
from functools import partial

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn import metrics
from pixel_cnn_pp import nn
from pixel_cnn_pp.model import model_spec
import data.cifar10_data as cifar10_data
import data.svhn_data as svhn_data


from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
print(get_available_gpus)

def compute_auc(neg, pos, pos_label=1):
    ys = np.concatenate((np.zeros(len(neg)), np.ones(len(pos))), axis=0)
    neg = np.array(neg)[np.logical_not(np.isnan(neg))]
    pos = np.array(pos)[np.logical_not(np.isnan(pos))]
    scores = np.concatenate((neg, pos), axis=0)
    auc = metrics.roc_auc_score(ys, scores)
    if pos_label == 1:
        return auc
    else:
        return 1 - auc

def compute_auc_llr(preds_in, preds_ood, preds0_in, preds0_ood):
    """Compute AUC for LLR."""
    # evaluate AUROC for OOD detection
    auc = compute_auc(
        preds_in, preds_ood, pos_label=0)
    llr_in = np.array(preds_in) - np.array(preds0_in)
    llr_ood = np.array(preds_ood) - np.array(preds0_ood)
    auc_llr = compute_auc(llr_in, llr_ood, pos_label=0)
    return auc, auc_llr

def compute_auc_grad(preds_in, preds_ood, preds0_in, preds0_ood):
    """Compute AUC for LLR."""

    # evaluate AUROC for OOD detection
    auc = compute_auc(
        preds_in, preds_ood, pos_label=0)
    llr_in = preds_in - preds0_in
    llr_ood = preds_ood - preds0_ood
    auc_llr = compute_auc(llr_in, llr_ood, pos_label=0)
    return auc, auc_llr

def print_and_write(fname, context):
    print(context + '\n')
    with open(fname, 'w') as f:
        f.write(context + '\n')

def get_complexity(args, dataset, eval_mode):
    """ Adapted from https://github.com/boschresearch/hierarchical_anomaly_detection/blob/master/SerraReplicationCode/ReferenceGlowVsDirectPng.py """
    data = get_data(args, dataset, eval_mode)
    all_bpds = []
    for batch in data:
        for a_x in batch:
            # Use highest compression level (9)
            img_encoded = cv2.imencode('.png', a_x, [int(cv2.IMWRITE_PNG_COMPRESSION),9])[1]
            assert img_encoded.shape[1] == 1
            all_bpds.append((len(img_encoded) * 8)/np.prod(a_x.shape))
    return all_bpds


parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('--ckpt_file', type=str, default='/local_home/tim/pxpp/data', help='path for file, e.g. /path/params_cifar.ckpt')
parser.add_argument('--in_data', type=str, default='cifar', help='cifar|cifar_samples')
parser.add_argument('--ood_data', type=str, default='svhn', help='svhn|celeba|cifar100')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5, help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160, help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10, help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='concat_elu', help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')
parser.add_argument('-c', '--class_conditional', dest='class_conditional', action='store_true', help='Condition generative model on labels?')
parser.add_argument('-ed', '--energy_distance', dest='energy_distance', action='store_true', help='use energy distance in place of likelihood')
parser.add_argument('-cl', '--continuous_logistic', dest='continuous_logistic', action='store_true', help='use logistic instead of discretized and bounded logistic')
# optimization
parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995, help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size during training per GPU')
parser.add_argument('-u', '--init_batch_size', type=int, default=16, help='How much data to use for data-dependent initialization.')
parser.add_argument('-p', '--dropout_p', type=float, default=0.5, help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int, default=5000, help='How many epochs to run in total?')
parser.add_argument('-g', '--nr_gpu', type=int, default=8, help='How many GPUs to distribute the training across?')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
parser.add_argument('-t', '--small_test', dest='small_test', action='store_true', help='test on small data')
args = parser.parse_args()

rng = np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)

if args.continuous_logistic:
    log_prob_func = nn.continuous_mix_logistic_loss
    cdf_func = nn.cdf_transform_continuous
else:
    log_prob_func = nn.discretized_mix_logistic_loss
    cdf_func = nn.cdf_transform_discretized


def get_log_probs(model, args, dataset, eval_mode):
    return get_preds(model, args, dataset, eval_mode, log_prob_func)

def get_cdf_transform(model, args, dataset, eval_mode):
    return get_preds(model, args, dataset, eval_mode, cdf_func)

def get_data(args, dataset, eval_mode):
    if dataset == 'cifar':
        DataLoader = cifar10_data.DataLoader
        data_dir = '../data/'
    elif dataset == 'svhn':
        DataLoader = svhn_data.DataLoader
        data_dir = '../data/svhn'
    elif dataset == 'cifar100':
        raise NotImplemented("TODO")
    elif dataset == 'celeba':
        raise NotImplemented("TODO")
    elif dataset == 'cifar_samples':
        raise NotImplemented("TODO")
    else:
        raise("unsupported dataset")
    data = DataLoader(data_dir, eval_mode, args.batch_size * args.nr_gpu, rng=rng, shuffle=False, return_labels=args.class_conditional, small_test=args.small_test)
    return data

def get_preds(model, args, dataset, eval_mode, func):
    data = get_data(args, dataset, eval_mode)
    obs_shape = data.get_observation_size()
    # data = (data.data.astype(np.float32) - 127.5) / 127.5
    # dataset = tf.data.Dataset.from_tensor_slices(data).batch(args.batch_size) # .map(lambda s: (tf.cast(s, tf.float32) - 127.5) / 127.5)
    # iterator = dataset.make_one_shot_iterator()
    # next_element = iterator.get_next()

    h_sample = [None] * args.nr_gpu
    hs = h_sample
    h_init = None

    xs = [tf.placeholder(tf.float32, shape=(args.batch_size, ) + obs_shape) for i in range(args.nr_gpu)]
    x_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + obs_shape)

    def make_feed_dict(data, obs_shape, init=False):
        if type(data) is tuple:
            x,y = data
        else:
            x = data
            y = None
        x = np.cast[np.float32]((x - 127.5) / 127.5) # input to pixelCNN is scaled from uint8 [0,255] to float in range [-1,1]
        if init:
            feed_dict = {x_init: x}
            if y is not None:
                feed_dict.update({y_init: y})
        # else:
            # x = np.split(x, args.nr_gpu)
            # feed_dict = {xs[i]: x[i] for i in range(args.nr_gpu)}
            # if y is not None:
            #     y = np.split(y, args.nr_gpu)
            #     feed_dict.update({ys[i]: y[i] for i in range(args.nr_gpu)})
        return feed_dict

    model_opt = { 'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters, 'nr_logistic_mix': args.nr_logistic_mix, 'resnet_nonlinearity': args.resnet_nonlinearity, 'energy_distance': args.energy_distance }
    train_losses = []
    # all_log_probs = []
    # all_ar_resids = []
    # all_cdf_transform = []
    i = 0
    init_pass = model(x_init, h_init, init=True, dropout_p=args.dropout_p, **model_opt)
    # out = model(next_element, h_init, init=True, dropout_p=0, **model_opt)
    initializer = tf.global_variables_initializer()
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement = True)
    with tf.Session(config=config) as sess:
        saver.restore(sess, args.ckpt_file)
        with tf.device('/gpu:%d' % i):
            # l = sess.run(func(next_element, out))
            # train_losses.append(l)
            for x in data:
                # feed_dict = make_feed_dict(d, obs_shape)
                x = np.cast[np.float32]((x - 127.5) / 127.5)
                feed_dict = {xs[i]: x}
                out = model(xs[i], hs[i], ema=None, dropout_p=0, **model_opt)
                # log_probs, ar_resids, cdf_transform = sess.run(func(xs[i], out, sum_all='pixel'), feed_dict)
                # all_log_probs.extend(log_probs)
                # all_ar_resids.extend(ar_resids)
                # all_cdf_transform.extend(cdf_transform)
                l = sess.run(func(xs[i], out, sum_all='image'), feed_dict)
                train_losses.extend(l)
    return train_losses
    # return all_log_probs, all_ar_resids, all_cdf_transform

def get_entropy(log_probs):
    return np.mean(log_probs)

def distance_from_unif(samples, test='ks'):
    sorted_samples = np.sort(samples, axis=1)
    assert (np.greater_equal(sorted_samples, 0)).all(), sorted_samples
    assert (np.less_equal(sorted_samples, 1)).all(), sorted_samples
    ts_test = partial(ts, test=test)
    return np.apply_along_axis(ts_test, 0, sorted_samples)

def ts(sorted_samples, test):
    n = len(sorted_samples)
    if test == 'ks':
        # should not include 0 but include 1
        unif_cdf = list(np.arange(0, 1, 1/n))[1:] + [1.0]
        return max(abs(sorted_samples - unif_cdf))
    elif test == 'cvm':
        ts = 1/(12 * n)
        for i in range(1, n + 1):
            ts += (sorted_samples[i-1] - (2*i - 1)/n)**2
        return ts
    elif test == 'ad':
        ts = 0
        for i in range(1, n + 1):
            ts += (2*i - 1) * math.log(np.maximum(sorted_samples[i-1], [1e-16]))
            ts += (2*n + 1 - 2*i) * math.log(np.maximum(1 - sorted_samples[i-1], [1e-16]))
        ts /= n
        ts -= n
        return ts


def time_series_test(
    inl_train_fea, inl_test_fea, oul_fea, test_typ, Bsz=1, Nsamples=None,
    L=100, SK=1):
    """
    From https://github.com/thu-ml/ood-dgm/blob/master/pixelcnn/ardgm_tests.ipynb
    :param Bsz: batch size in a multi-sample test (as in the multi-sample typicality test, arXiv:1906.02994).
    :param SK: only include lags which are multiples of SK in the test
    :param L: the maximum lag to use
    """
    oul_fea = normalize_feature(inl_train_fea, oul_fea)
    inl_fea = normalize_feature(inl_train_fea, inl_test_fea)
    oul_stats, _ = get_autocorr(oul_fea, Nsamples, L, test_typ, SK)
    inl_stats, _ = get_autocorr(inl_fea, Nsamples, L, test_typ, SK)
    return inl_stats, oul_stats


def normalize_feature(inl_train, ood, batch_dims=1):
    inl_train = inl_train.reshape([np.prod(inl_train.shape[:batch_dims]), -1])
    ood = ood.reshape([np.prod(ood.shape[:batch_dims]), -1])
    _mean = inl_train.mean(axis=0)[None]
    _sd = inl_train.std(axis=0)[None]
    return (ood - _mean) / _sd

def autocorr5(x, lags):
    '''
    adapted from https://stackoverflow.com/a/51168178/7509266
    Fixed the incorrect denominator: np.correlate(x,x,'full')[d-1+k] returns
        sum_{i=k}^{d-1} x[i]x[i-k]
    so we should divide it by (d-k) instead of d
    '''
    mean=x.mean()
    var=np.var(x)
    xp=x-mean
    ruler = len(x) - np.arange(len(x))
    corr=np.correlate(xp,xp,'full')[len(x)-1:]/var/ruler
    return corr[:(lags)]

def get_autocorr(fea, B=None, L=200, test_typ='bp', skip_corr=1):
    idcs = np.arange(fea.shape[0])
    if B is not None:
        np.random.shuffle(idcs)
    else:
        B = idcs.shape[0]
    corrs = []
    N = fea[0].shape[0]
    for j in range(B):
        ac_j = autocorr5(fea[idcs[j]], L)
        corrs.append(ac_j)
    corrs = np.array(corrs)[:,::skip_corr]
    if test_typ == 'ljb':
        ruler = (N - np.arange(1, L+1))[None, ::skip_corr].astype('f')
        stats = N * (N+2) * (corrs[:,1:]**2 / ruler[:,1:]).mean(axis=-1)  # normalized; *L would follow ChiSq[L]
    else:
        stats = N * (corrs[:,1:]**2).astype('f').mean(axis=-1)
    return stats, corrs

def main():
    # # write results to file
    # out_dir = os.path.join(args.exp + '_save_dir', 'results')
    # from pathlib import Path
    # Path(out_dir).mkdir(exist_ok=True)
    # out_f = os.path.join(out_dir, 'run%s.txt' % args.suffix)
    # load model
    tf.logging.log(tf.logging.INFO, 'starting the run')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    tf.get_logger().setLevel('INFO')
    tf.logging.set_verbosity(tf.logging.INFO)
    name = "_".join([args.in_data, args.ood_data, os.path.dirname(args.ckpt_file)])
    model = tf.make_template('model', model_spec)
    tf.logging.log(tf.logging.INFO, 'initializing')
    initializer = tf.global_variables_initializer()
    tf.logging.log(tf.logging.INFO, 'initialized model')
    # LR
    # log_probs_pixel_in = get_log_probs(model, args, args.in_data, 'test')  # (N,32,32,3)
    # log_probs_in = np.mean(log_probs_pixel_in, axis=(1, 2, 3))
    # np.save('intermediate/' + name + '_log_probs_in.npy', log_probs_in)
    # log_probs_pixel_ood = get_log_probs(model, args, args.ood_data, 'test')
    # log_probs_ood = np.mean(log_probs_pixel_ood, axis=(1, 2, 3))
    # np.save('intermediate/' + name + '_log_probs_ood.npy', log_probs_ood)
    # complexity_in = get_complexity(args, args.in_data, 'test')
    # complexity_ood = get_complexity(args, args.ood_data, 'test')
    # print(log_probs_in, log_probs_ood)
    # print(len(log_probs_in),len(log_probs_ood),len(complexity_in), len(complexity_ood))
    # auc, auc_llr = compute_auc_llr(log_probs_in, log_probs_ood, complexity_in, complexity_ood)
    # tf.logging.log(tf.logging.INFO, f'LL: {auc}')
    # tf.logging.log(tf.logging.INFO, f'LR: {auc_llr}')
    # with open(f'results/{name}.txt', 'a') as f:
    #     f.write(f'LL: {auc}\n')
    #     f.write(f'LR: {auc_llr}\n')
    # # TT
    # log_probs_pixel_train = get_log_probs(model, args, args.in_data, 'train')
    # log_probs_train = np.mean(log_probs_pixel_train, axis=(1, 2, 3))
    # np.save('intermediate/' + name + '_log_probs_train.npy', log_probs_train)
    # train_entropy = get_entropy(log_probs_train)
    # typical_ts_in = list(map(abs, log_probs_in - train_entropy))
    # typical_ts_ood = list(map(abs, log_probs_ood - train_entropy))
    # print('before', typical_ts_in, typical_ts_ood)
    # # want higher to be better
    # print('-1', np.array(typical_ts_in) * -1, np.array(typical_ts_ood) * -1)
    # auc_tt = compute_auc(np.array(typical_ts_in) * -1, np.array(typical_ts_ood) * -1)
    # tf.logging.log(tf.logging.INFO, f'TT: {auc_tt}')
    # with open(f'results/{name}.txt', 'a') as f:
    #     f.write(f'TT: {auc_tt}')
    # # WN
    # wn_in, wn_ood = time_series_test(np.array(log_probs_pixel_train), np.array(log_probs_pixel_in), np.array(log_probs_pixel_ood), 'bp')
    # print(len(wn_in), len(wn_ood))
    # auc_wn = compute_auc(wn_in * -1, wn_ood * -1)
    # print(f'WN: {auc_wn}')
    # with open(f'results/{name}.txt', 'a') as f:
    #     f.write(f'UNIF: {auc_wn}')
    # UNIF
    unifs_in = get_cdf_transform(model, args, args.in_data, 'test')  # (B,32,32,3)
    unifs_ood = get_cdf_transform(model, args, args.ood_data, 'test')
    # for metric in ['ks', 'cvm', 'ad']:
    for metric in ['ad']:
        gof_ts_in = distance_from_unif(unifs_in, metric).flatten()
        gof_ts_ood = distance_from_unif(unifs_ood, metric).flatten()
        print(len(gof_ts_in), len(gof_ts_ood))
        # want higher to be better
        auc_unif = compute_auc(gof_ts_in * -1, gof_ts_ood * -1)
        print(f'UNIF: {auc_unif}')
        with open(f'results/{name}.txt', 'a') as f:
            f.write(f'UNIF {metric}: {auc_unif}')


if __name__ == '__main__':
  main()
