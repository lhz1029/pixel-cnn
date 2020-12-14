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

from absl import app
from absl import flags
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow.compat.v1 as tf
from sklearn import metrics
from pixel_cnn_pp import nn


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

def print_and_write(f, context):
  print(context + '\n')
  f.write(context + '\n')

def get_complexity(exp, data_dir, eval_mode_in, eval_mode_ood):
    if eval_mode_in == 'tr':
        eval_mode_in = 'train'
    if eval_mode_ood == 'tr':
        eval_mode_ood = 'train'
    if exp == 'fashion':
        in_data_name = 'fashion_mnist'
        ood_data_name = 'mnist'
    elif exp == 'mnist':
        in_data_name = 'mnist'
        ood_data_name = 'fashion_mnist'
    elif exp == 'mnist':
        in_data_name = 'mnist'
        ood_data_name = 'fashion_mnist'
    elif exp == 'mnist':
        in_data_name = 'mnist'
        ood_data_name = 'fashion_mnist'
    else:
        raise ValueError("exp complexity not supported: ", exp)
    data_in_path = os.path.join(data_dir, in_data_name + '_' + eval_mode_in, '*.png')
    data_ood_path = os.path.join(data_dir, ood_data_name + '_' + eval_mode_ood, '*.png')
    in_fnames = sorted(glob.glob(data_in_path))
    ood_fnames = sorted(glob.glob(data_ood_path))
    preds0_in_bits = [os.stat(fname).st_size * 8 for fname in in_fnames]
    preds0_ood_bits = [os.stat(fname).st_size * 8 for fname in ood_fnames]
    preds0_in = {}
    preds0_ood = {}
    preds0_in['labels'] = [math.log(2 ** bits) for bits in preds0_in_bits]
    preds0_ood['labels'] = [math.log(2 ** bits) for bits in preds0_ood_bits]
    preds0_in['log_probs'] = [math.log(2 ** bits) for bits in preds0_in_bits]
    preds0_ood['log_probs'] = [math.log(2 ** bits) for bits in preds0_ood_bits]
    return preds0_in, preds0_ood

# def plot_heatmap(n, data, plt_file, colorbar=True):
#   """Plot heatmaps (Figure 3 in the paper)."""
#   sns.set_style('whitegrid')
#   sns.set(style='ticks', rc={'lines.linewidth': 4})
#   cmap_reversed = ListedColormap(sns.color_palette('Greys_r', 6).as_hex())
#   fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(2 * n - 2, 2 * n - 2))
#   i = 0
#   for ax in axes.flat:
#     im = ax.imshow(data[i], vmin=0, vmax=6, cmap=cmap_reversed)
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     i += 1
#   fig.subplots_adjust(right=0.9)
#   if colorbar:
#     cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
#     fig.colorbar(im, cax=cbar_ax)
#     cbar_ax.tick_params(labelsize=20)

#   with tf.gfile.Open(plt_file, 'wb') as sp:
#     plt.savefig(sp, format='pdf', bbox_inches='tight')

# def calculate_zeros(exp, data_dir):
#   if exp == 'fashion':
#     test_in = os.path.join(data_dir, 'fashion_mnist_test.npy')
#     test_ood = os.path.join(data_dir, 'mnist_test.npy')
#   elif exp == 'mnist':
#     test_in = os.path.join(data_dir, 'mnist_test.npy')
#     test_ood = os.path.join(data_dir, 'fashion_mnist_test.npy')
#   else:
#     raise ValueError("exp not supported: ", exp)
#   img_in = np.load(test_in)
#   img_ood = np.load(test_ood)
#   img_in = img_in.reshape((img_in.shape[0], -1))
#   img_ood = img_ood.reshape((img_ood.shape[0], -1))
#   # zeros_in = (img_in == 0).sum(axis=1) / img_in.shape[1]
#   # zeros_ood = (img_ood == 0).sum(axis=1) / img_ood.shape[1]
#   zeros_in = np.mean(img_in, axis=1)
#   zeros_ood = np.mean(img_in, axis=1)
#   return zeros_in, zeros_ood

import argparse
from pixel_cnn_pp.model import model_spec
import data.cifar10_data as cifar10_data
import data.svhn_data as svhn_data

parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str, default='/local_home/tim/pxpp/data', help='Location for the dataset')
parser.add_argument('--ckpt_file', type=str, default='/local_home/tim/pxpp/data', help='path for file, e.g. /path/params_cifar.ckpt')
parser.add_argument('--exp', type=str, default='cifar', help='cifar|svhn')
parser.add_argument('--suffix', type=str, default='', help='suffix for results file')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5, help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160, help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10, help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='concat_elu', help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')
parser.add_argument('-c', '--class_conditional', dest='class_conditional', action='store_true', help='Condition generative model on labels?')
parser.add_argument('-ed', '--energy_distance', dest='energy_distance', action='store_true', help='use energy distance in place of likelihood')
parser.add_argument('-ed', '--deriv_constraint', dest='deriv_constraint', action='store_true', help='use derivative constraint (only for likelihood)')
# optimization
parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995, help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size during training per GPU')
parser.add_argument('-u', '--init_batch_size', type=int, default=16, help='How much data to use for data-dependent initialization.')
parser.add_argument('-p', '--dropout_p', type=float, default=0.5, help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int, default=5000, help='How many epochs to run in total?')
parser.add_argument('-g', '--nr_gpu', type=int, default=8, help='How many GPUs to distribute the training across?')
# evaluation
parser.add_argument('--polyak_decay', type=float, default=0.9995, help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
parser.add_argument('-ns', '--num_samples', type=int, default=1, help='How many batches of samples to output.')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
args = parser.parse_args()

rng = np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)

def get_preds(sess, model, args, eval_mode_in, eval_mode_ood):
    # get data
    if args.exp == 'cifar':
        TrainDataLoader = cifar10_data.DataLoader
        TestDataLoader = svhn_data.DataLoader
    elif args.exp == 'svhn':
        TrainDataLoader = svhn_data.DataLoader
        TestDataLoader = cifar10_data.DataLoader
    else:
        raise("unsupported dataset")
    train_data = TrainDataLoader(args.data_dir, eval_mode_in, args.batch_size * args.nr_gpu, rng=rng, shuffle=True, return_labels=args.class_conditional)
    test_data = TestDataLoader(args.data_dir, eval_mode_ood, args.batch_size * args.nr_gpu, shuffle=False, return_labels=args.class_conditional)
    obs_shape = train_data.get_observation_size()

    h_sample = [None] * args.nr_gpu
    hs = h_sample

    xs = [tf.placeholder(tf.float32, shape=(args.batch_size, ) + obs_shape) for i in range(args.nr_gpu)]

    def make_feed_dict(data, obs_shape, init=False):
        x_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + obs_shape)
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
    test_losses = []
    for i in range(args.nr_gpu):
        with tf.device('/gpu:%d' % i):
            for d in test_data:
                feed_dict = make_feed_dict(d, obs_shape)
                out = model(xs[i], hs[i], ema=None, dropout_p=0, **model_opt)
                loss_fun = nn.discretized_mix_logistic_loss
                l = sess.run(loss_fun(xs[i], out), feed_dict)
                test_losses.append(l)
            for d in train_data:
                feed_dict = make_feed_dict(d, obs_shape)
                out = model(xs[i], hs[i], ema=None, dropout_p=0, **model_opt)
                loss_fun = nn.discretized_mix_logistic_loss
                l = sess.run(loss_fun(xs[i], out), feed_dict)
                train_losses.append(l)
    return train_losses, test_losses

def main(unused_argv):
    # write results to file
    out_dir = os.path.join(args.exp + '_save_dir', 'results')
    tf.compat.v1.gfile.MakeDirs(out_dir)
    out_f = tf.compat.v1.gfile.Open(
        os.path.join(out_dir, 'run%d.txt' % args.suffix), 'w')
    # load model
    model = tf.make_template('model', model_spec)
    initializer = tf.global_variables_initializer()
    saver = tf.train.Saver()
    ckpt_file = args.ckpt_file
    print('restoring parameters from', ckpt_file)
    # get preds
    with tf.Session() as sess:
        saver.restore(sess, ckpt_file)
        preds_in, preds_ood = get_preds(sess, model, args, 'test', 'test')

    preds0_in, preds0_ood = get_complexity(args.exp, args.data_dir, 'test', 'test')
    auc, auc_llr = compute_auc_llr(preds_in, preds_ood, preds0_in, preds0_ood)
    zeros_in = preds0_in
    zeros_ood = preds0_ood
    # plot
    plt.scatter(zeros_in, preds_in, color='blue', alpha=.2)
    plt.scatter(zeros_ood, preds_ood, color='red', alpha=.2)
    plt.title(args.exp + ' likelihood')
    plt.savefig(args.exp + ' likelihood' + '.pdf', bbox_inches='tight')
    plt.clf()
    plt.scatter(zeros_in, preds_in, color='blue', alpha=.2)
    plt.scatter(zeros_ood, preds_ood, color='red', alpha=.2)
    plt.title(args.exp + ' likelihood')
    plt.savefig(args.exp + ' likelihood' + '.pdf', bbox_inches='tight')
    plt.clf()
    print_and_write(out_f, 'final test, auc={}, auc_llr={}'.format(auc, auc_llr))



  

  ## Final test on FashionMNIST-MNIST/CIFAR-SVHN
  # foreground model
  preds_in, preds_ood, grad_in, grad_ood = load_data_and_model_and_pred(
      FLAGS.exp,
      FLAGS.data_dir,
      0.0,
      0.0,
      FLAGS.repeat_id,
      FLAGS.ckpt_step,
      'test',
      'test',
      return_per_pixel=True)

  preds0_in, preds0_ood = get_complexity(FLAGS.exp, FLAGS.data_dir, 'test', 'test')
  auc, auc_llr = compute_auc_llr(preds_in, preds_ood, preds0_in, preds0_ood)
  zeros_in = preds0_in['log_probs']
  zeros_ood = preds0_ood['log_probs']
  plt.scatter(zeros_in, preds_in['log_probs'], color='blue', alpha=.2)
  plt.scatter(zeros_ood, preds_ood['log_probs'], color='red', alpha=.2)
  plt.title(FLAGS.exp + ' likelihood')
  plt.savefig(FLAGS.exp + ' likelihood' + '.pdf', bbox_inches='tight')
  plt.clf()
  print_and_write(out_f, 'final test, auc={}'.format(auc))

  # typicality approximation
  grad_in = grad_in.reshape((-1))
  grad_ood = grad_ood.reshape((-1))
  grad_auc = utils.compute_auc(
      grad_in, grad_ood, pos_label=0)
  print(zeros_in.shape, grad_in.shape)
  plt.scatter(zeros_in, grad_in, color='blue', alpha=.2)
  plt.scatter(zeros_ood, grad_ood, color='red', alpha=.2)
  plt.title(FLAGS.exp + ' typicality')
  plt.savefig(FLAGS.exp + ' typicality' + '.pdf', bbox_inches='tight')
  plt.clf()
  print_and_write(out_f, 'final test grad, auc={}'.format(grad_auc))

  out_f.close()

  # plot heatmaps (Figure 3)
  if FLAGS.exp in ['fashion', 'mnist']:
    n = 4

    # FashionMNIST
    log_probs_in = preds_in['log_probs']
    log_probs_pp_in = preds_in['log_probs_per_pixel']
    n_sample_in = len(log_probs_in)
    log_probs_in_sorted = sorted(
        range(n_sample_in), key=lambda k: log_probs_in[k], reverse=True)
    ids_seq = np.arange(1, n_sample_in, int(n_sample_in / (n * n)))

    ## pure likelihood
    data = [
        log_probs_pp_in[log_probs_in_sorted[ids_seq[i]]] + 6
        for i in range(n * n)
    ]
    plt_file = os.path.join(
        out_dir, f'run%d_heatmap_{FLAGS.exp}_test_in_p(x).pdf' % FLAGS.repeat_id)
    plot_heatmap(n, data, plt_file)

    # MNIST
    log_probs_ood = preds_ood['log_probs']
    log_probs_pp_ood = preds_ood['log_probs_per_pixel']
    n_sample_ood = len(log_probs_ood)
    log_probs_ood_sorted = sorted(
        range(n_sample_ood), key=lambda k: log_probs_ood[k], reverse=True)
    ids_seq = np.arange(1, n_sample_ood, int(n_sample_ood / (n * n)))

    ## pure likelihood
    data = [
        log_probs_pp_ood[log_probs_ood_sorted[ids_seq[i]]] + 6
        for i in range(n * n)
    ]
    plt_file = os.path.join(out_dir,
                            f'run%d_heatmap_{FLAGS.exp}_test_ood_p(x).pdf' % FLAGS.repeat_id)
    plot_heatmap(n, data, plt_file)


if __name__ == '__main__':
  app.run(main)
