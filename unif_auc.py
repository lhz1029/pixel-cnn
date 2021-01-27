import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import metrics
import argparse
from functools import partial

def distance_from_unif(samples, test='ks'):
    sorted_samples = np.sort(samples, axis=1)
    try:
        assert (np.greater_equal(sorted_samples, 0)).all(), np.min(sorted_samples)
        assert (np.less_equal(sorted_samples, 1)).all(), np.max(sorted_samples)
    except AssertionError:
        sorted_samples = np.maximum(sorted_samples, 0)
        sorted_samples = np.minimum(sorted_samples, 1)
    ts_test = partial(ts, test=test)
    return np.apply_along_axis(ts_test, 1, sorted_samples)

def ts(sorted_samples, test):
    n = len(sorted_samples)
    if test == 'ks':
        # should not include 0 but include 1
        unif_cdf = list(np.arange(0, 1, 1/n))[1:] + [1.0]
        return max(abs(sorted_samples - unif_cdf))
    elif test == 'cvm':
        # ts = 1/(12 * n)
        # for i in range(1, n + 1):
        #     ts += (sorted_samples[i-1] - (2*i - 1)/n)**2
        # return ts
        return np.sum(np.square(np.array([(2*i - 1)/n for i in range(n)]) - sorted_samples)) + 1/(12*n)
    elif test == 'ad':
        # ts = 0
        # for i in range(1, n + 1):
        #     ts -= (2*i - 1) * math.log(np.maximum(sorted_samples[i-1], [1e-16]))
        #     ts -= (2*n + 1 - 2*i) * math.log(np.maximum(1 - sorted_samples[i-1], [1e-16]))
        # ts /= n
        # ts -= n
        # return ts
        Ws = np.array([(2*i - 1) for i in range(n)]) * np.log(np.maximum(sorted_samples, [1e-16]))
        Vs = np.array([(2*n + 1 - 2*i) for i in range(n)]) * np.log(np.maximum(1 - sorted_samples, [1e-16]))
        return (-np.sum(Ws) - np.sum(Vs))/n - n

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, default='',
                        help="Location of checkpoint to restore")
    parser.add_argument("-o", type=str, default='',
                        help="Location of checkpoint to restore")
    args = parser.parse_args()
    in_samples = np.load(args.i)
    out_samples = np.load(args.o)
    if len(in_samples.shape) > 2:
        in_samples = in_samples.reshape((in_samples.shape[0], -1))
        out_samples = out_samples.reshape((out_samples.shape[0], -1))
    # in_samples = np.random.uniform(size=(20, 3072))
    # out_samples = np.random.beta(a=1, b=1.5, size=(20, 3072))
    # for test in ['ks', 'cvm', 'ad']:
    for test in ['ad']:
        in_d = distance_from_unif(in_samples, test)
        print(np.min(in_d), np.max(in_d))
        out_d = distance_from_unif(out_samples, test)
        print(np.min(out_d), np.max(out_d))
        auc_unif = compute_auc(out_d * -1, in_d * -1)
        print(f'UNIF: {auc_unif}')