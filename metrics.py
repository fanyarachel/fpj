import numpy as np


def mean_reciprocal_rank(rank_true, rank_pred):
    """Mean Reciprocal Rank"""
    lst = []
    for p in rank_true:
        rank_true_order, rank_pred_order = list(zip(*rank_true[p]))[0], list(zip(*rank_pred[p]))[0]
        lst.append(1.0 / (rank_pred_order.index(rank_true_order[0]) + 1.0))
    return np.mean(lst)


def precision_at_k(r, k):
    """Precision @ k"""
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    """Average Precision (area under PR curve)"""
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rank_true, rank_pred):
    """Mean Average Precision"""
    lst = []
    for p in rank_true:
        rank_pred_gt = []
        for n, val in rank_pred[p]:
            n_idx = list(zip(*rank_true[p]))[0].index(n)
            if rank_true[p][n_idx][1] > 0:
                rank_pred_gt.append(1)
            else:
                rank_pred_gt.append(0)
        lst.append(average_precision(rank_pred_gt))
    return np.mean([lst])


def dcg_at_k(r, k, method=0):
    """Discounted Cumulative Gain"""
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Normalized Discounted Cumulative Gain (NDCG)"""
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def mean_ndcg_at_k(rank_true, rank_pred, k):
    """Mean NDCG@k"""
    lst = []
    for p in rank_true:
        rank_pred_gt = []
        for n, val in rank_pred[p]:
            n_idx = list(zip(*rank_true[p]))[0].index(n)
            rank_pred_gt.append(rank_true[p][n_idx][1])
        lst.append(ndcg_at_k(rank_pred_gt, k))
    return np.mean(lst)
