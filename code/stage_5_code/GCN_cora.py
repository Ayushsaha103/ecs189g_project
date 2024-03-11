import torch
import torch.nn as nn
import torch.nn.functional as F
from .pygcn.layers import GraphConvolution

class GCN_cora(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout_rate):
        super(GCN_cora, self).__init__()

        # First graph convolution layer
        self.gc1 = GraphConvolution(nfeat, nhid)

        # Second layer (outputs the class scores)
        self.gc2 = GraphConvolution(nhid, nclass)

        # Dropout rate
        self.dropout_rate = dropout_rate

    def forward(self, x, adj):
        # Input x shape: [N, nfeat], adj shape: [N, N]

        # Apply first graph convolution layer
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout_rate, training=self.training)

        # Apply second layer
        x = self.gc2(x, adj)  # Output raw class scores

        # Apply log softmax to get log probabilities
        return F.log_softmax(x, dim=1)
