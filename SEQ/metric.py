import numpy as np
import math


def precision_recall(topk_items, test_items):
    """
    Input:
        topk_items: list of top-K recommendation
        test_items: list of eval items.
        
    Return:
        tuple (precision, recall, NDCG)
    """
    if len(topk_items) != len(test_items):
        raise RuntimeError("Must have the same length.")
        
    precision = recall = dcg = 0.0
    ln2 = math.log(2)
    
    for topk, test in zip(topk_items, test_items):
        hits = np.intersect1d(topk, test)
        precision += len(hits) / len(topk)
        recall    += len(hits) / len(test)
        
        rank = np.argwhere(np.in1d(topk, test)) + 1  # The rank of items that topk hit.
        dcg += np.sum( ln2 / np.log1p(rank) )
        
    n = len(topk_items)
    
    return precision / n, recall / n, dcg / n