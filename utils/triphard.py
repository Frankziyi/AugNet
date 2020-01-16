import torch
import torch.nn as nn
import pdb

class TripHard(nn.Module):
    def __init__(self, margin=0.3):
        super(TripHard, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.TripletMarginLoss(margin=margin)

    def forward(self, inputs, labels):
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n,n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()

        mask = labels.expand(n, n).eq(labels.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

class UnsupervisedTriphard(nn.Module):
    def __init__(self, margin=0.3):
        super(UnsupervisedTriphard, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, inputs, positive):
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n,n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()

        negative = []
        for i in range(n):
            negative.append(inputs[dist[i].argsort()[2]]) #???
        
        negative = torch.stack(negative)
        loss = self.ranking_loss(inputs, positive, negative)
        return loss
