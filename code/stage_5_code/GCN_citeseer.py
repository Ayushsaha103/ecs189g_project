import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("../")
from pygcn.layers import GraphConvolution


class GCN_citeseer(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_citeseer, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)