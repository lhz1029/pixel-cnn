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
import glob
import math
import argparse
import cv2

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow.compat.v1 as tf
from sklearn import metrics
from pixel_cnn_pp import nn
from pixel_cnn_pp.model import model_spec
import data.cifar10_data as cifar10_data
import data.svhn_data as svhn_data


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
  llr_in = preds_in - preds0_in
  llr_ood = preds_ood - preds0_ood
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
    for a_x in data:
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
parser.add_argument('-g', '--nr_gpu', type=int, default=8, help='How many GPUs to distribute the training across?')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
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
    data = DataLoader(data_dir, eval_mode, args.batch_size * args.nr_gpu, rng=rng, shuffle=False, return_labels=args.class_conditional)
    return data

def get_preds(model, args, dataset, eval_mode, func):
    data = get_data(args, dataset, eval_mode)
    obs_shape = data.get_observation_size()

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
        else:
            x = np.split(x, args.nr_gpu)
            feed_dict = {xs[i]: x[i] for i in range(args.nr_gpu)}
            # if y is not None:
            #     y = np.split(y, args.nr_gpu)
            #     feed_dict.update({ys[i]: y[i] for i in range(args.nr_gpu)})
        return feed_dict

    model_opt = { 'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters, 'nr_logistic_mix': args.nr_logistic_mix, 'resnet_nonlinearity': args.resnet_nonlinearity, 'energy_distance': args.energy_distance }
    train_losses = []
    i = 0
    init_pass = model(x_init, h_init, init=True, dropout_p=args.dropout_p, **model_opt)
    initializer = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, args.ckpt_file)
        with tf.device('/gpu:%d' % i):
            for d in data:
                feed_dict = make_feed_dict(d, obs_shape)
                out = model(xs[i], hs[i], ema=None, dropout_p=0, **model_opt)
                l = sess.run(func(xs[i], out), feed_dict)
                train_losses.append(l)
    return train_losses

def get_entropy(log_probs):
    return tf.math.reduce_mean(log_probs)

def distance_from_unif(samples, test='ks'):
    n = len(samples)
    sorted_samples = sorted(samples)
    assert all([0 <= s <= 1 for s in sorted_samples])
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
            ts += (2*i - 1) * math.log(sorted_samples[i-1])
            ts += (2*n + 1 - 2*i) * math.log(1 - sorted_samples[i-1])
        ts /= n
        ts -= n
        return ts

def main():
    # # write results to file
    # out_dir = os.path.join(args.exp + '_save_dir', 'results')
    # from pathlib import Path
    # Path(out_dir).mkdir(exist_ok=True)
    # out_f = os.path.join(out_dir, 'run%s.txt' % args.suffix)
    # load model
    model = tf.make_template('model', model_spec)
    initializer = tf.global_variables_initializer()
    # LR
    log_probs_in = get_log_probs(model, args, args.in_data, 'test')
    log_probs_ood = get_log_probs(model, args, args.ood_data, 'test')
    complexity_in = get_complexity(args, args.in_data, 'test')
    complexity_ood = get_complexity(args, args.ood_data, 'test')
    auc, auc_llr = compute_auc_llr(log_probs_in, log_probs_ood, complexity_in, complexity_ood)
    print(f'LL: {auc}')
    print(f'LR: {auc_llr}')
    # TT
    train_log_probs = get_log_probs(model, args, args.in_data, 'train')
    train_entropy = get_entropy(train_log_probs)  # TODO
    typical_ts_in = list(map(abs, log_probs_in - train_entropy))
    typical_ts_ood = list(map(abs, log_probs_ood - train_entropy))
    # want higher to be better
    auc_tt = compute_auc(typical_ts_in * -1, typical_ts_ood * -1)
    print(f'TT: {auc_tt}')
    # WN
    # TODO https://github.com/thu-ml/ood-dgm/blob/master/pixelcnn/ardgm_tests.ipynb

    # UNIF
    unifs_in = get_cdf_transform(model, args, args.in_data, 'test')
    unifs_ood = get_cdf_transform(model, args, args.ood_data, 'test')
    gof_ts_in = distance_from_unif(unifs_in, 'ks')
    gof_ts_ood = distance_from_unif(unifs_ood, 'ks')
    # want higher to be better
    auc_unif = compute_auc(typical_ts_in * -1, typical_ts_ood * -1)
    print(f'UNIF: {auc_unif}')


if __name__ == '__main__':
  main()
