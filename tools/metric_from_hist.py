import numpy as np
from datetime import datetime

def read_hist_from_file(fname):
    hist = np.loadtxt(fname)
    return hist

def compute_mean_metrics(hist, iter):
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'overall accuracy', acc
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    print '>>>', datetime.now(), 'Iteration', iter, 'mean accuracy', np.nanmean(acc)
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>', datetime.now(), 'Iteration', iter, 'mean IU', np.nanmean(iu)
    freq = hist.sum(1) / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'fwavacc', \
            (freq[freq > 0] * iu[freq > 0]).sum()
    # per-class Dice Score / F1, dice = 2*iu/(1+iu)
    dice = 2 * np.diag(hist) / (hist.sum(1) + hist.sum(0))
    print '>>>', datetime.now(), 'Iteration', iter, 'mean Dice Score', \
            np.nanmean(dice)

def main():
    hist_file = 'data/VOCdevkit2007/VOC2007/hist.txt'
    hist = read_hist_from_file(hist_file)
    compute_mean_metrics(hist, 0)


main()
