# Code taken from https://gist.github.com/bwhite/3726239
import numpy as np


def dcg(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.


def ndcg(r, k):
    idcg = dcg(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg(r, k) / idcg
