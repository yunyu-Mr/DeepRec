import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from metric import precision_recall
from spotlight.layers import ScaledEmbedding, ZeroEmbedding


def predict(net, sequences, n_items, item_ids=None):
    """
    net: model
    sequences: 2D array
    item_ids: 2D array
    """
    # Set to test mode (will not dropout or batch norm)
    net.train(False)
    
    sequences = np.atleast_2d(sequences)
    
    if item_ids is None:
        item_ids = np.atleast_2d(np.arange(n_items))
        item_ids = item_ids.repeat(len(sequences), axis=0)
    else:
        item_ids = np.atleast_2d(item_ids)
        assert(len(sequences) == len(item_ids))
    
    n_items = item_ids.shape[1]
    
    # To tensor
    sequences = torch.from_numpy(sequences.astype('int64'))
    item_ids = torch.from_numpy(item_ids.astype('int64'))
    
    # To variable
    sequence_var = Variable(sequences)
    item_var = Variable(item_ids)
    
    # Get user representation
    _, user_final = net.user_representation(sequence_var)
    
    shape = list(user_final.size())  # (N, E, 1)
    shape[2] = n_items
    user_final = user_final.expand(shape)  # (N, E, L)
    
    # Prediction
    out = net(user_final, item_var)
    
    return out.data.numpy()


def evaluate(net, sequences, eval_sequences, n_items, n_top=20):
    """
    Recommend top N items.
    
    Input:
        net: nn.Module
        sequences: padded item sequences
        eval_sequences: sequences for evaluation
        n_top: top N
    
    Return:
        Precision@N
        Recall@N
        NDCG@N
    """
    output = predict(net, sequences, n_items)
    topk_recs = np.argsort(-output)[:, :n_top]
    return precision_recall(topk_recs, eval_sequences)