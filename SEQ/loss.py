import torch
import torch.nn.functional as F


def bpr_loss(positive_predictions, negative_predictions, mask=None):
    """
    Bayesian Personalised Ranking [1]_ pairwise loss function.
    """
    loss = - F.logsigmoid(positive_predictions - negative_predictions)
    
    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()

    return loss.mean()

def bpr2_loss(positive_predictions, negative_predictions, mask=None):
    """
    Bayesian Personalised Ranking [1]_ pairwise loss function.
    """
    loss = (1.0 - F.sigmoid(positive_predictions -
                            negative_predictions))
    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()

    return loss.mean()


def top1_loss(positive_predictions, negative_predictions, mask=None):
    """
    TOP1 loss
    """
    loss = F.sigmoid(negative_predictions - positive_predictions) \
         + F.sigmoid(negative_predictions**2)
    
    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()

    return loss.mean()


def hinge_loss(positive_predictions, negative_predictions, mask=None):
    """
    SVM like hinge loss.
    """
    loss = torch.clamp(negative_predictions -
                       positive_predictions +
                       1.0, 0.0)
    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()

    return loss.mean()