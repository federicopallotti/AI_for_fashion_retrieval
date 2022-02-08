import torch
from torch import nn
import torch.nn.functional as F

#THIS IMPLEMENTATION OF TRIPLET LOSS IS PROPERTY OF AIMAGELAB, MODENA

def l2norm(X, dim, eps=1e-8):
    norm = (torch.pow(X, 2).sum(dim=dim, keepdim=True) + eps).sqrt()
    X = torch.div(X, norm)
    return X


class TripletLoss(nn.Module):
    def __init__(self, margin=0., max_violation=False, reduction='sum'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation
        self.reduction = reduction

    def forward(self, scores):
        diagonal = scores.diag().view(scores.size(0), 1)
        d = diagonal.expand_as(scores)

        # compare every diagonal score to scores in its column
        cost = (self.margin + scores - d).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = mask.to(scores.device)
        cost = cost.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost = cost.max(1)[0]

        if self.reduction == 'sum':
            return cost.sum()
        elif self.reduction == 'mean':
            return cost.mean()
        else:
            raise NotImplementedError


def constrastiveLoss(x1, x2, label):
  euclidean_distance = F.pairwise_distance(x1, x2)
  loss = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                      (label) * torch.pow(torch.clamp(2.- euclidean_distance, min=0.0), 2))
  return loss