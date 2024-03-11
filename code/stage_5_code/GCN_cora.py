import torch
import torch.nn as nn
import torch.nn.functional as F
from .pygcn.layers import GraphConvolution

class GCN_cora(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout_rate):
        super(GCN_cora, self).__init__()

        # First graph convolution layer
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)  # Batch normalization layer for layer outputs

        # Second layer
        self.gc2 = GraphConvolution(nhid, nhid)
        self.bn2 = nn.BatchNorm1d(nhid)  # Batch normalization layer for layer outputs

        # Third layer (outputs the class scores)
        self.gc3 = GraphConvolution(nhid, nclass)

        self.dropout_rate = dropout_rate

    def forward(self, x, adj):
        # Input x shape: [N, nfeat], adj shape: [N, N]

        # Apply first graph convolution layer
        x = F.relu(self.bn1(self.gc1(x, adj)))  # Apply batch normalization before ReLU
        x = F.dropout(x, self.dropout_rate, training=self.training)  # Apply dropout

        # Apply second layer
        x = F.relu(self.bn2(self.gc2(x, adj)))  # Apply batch normalization before ReLU
        x = F.dropout(x, self.dropout_rate, training=self.training)  # Apply dropout

        # Apply third layer
        x = self.gc3(x, adj)  # Output raw class scores

        # Apply log softmax to get log probabilities
        return F.log_softmax(x, dim=1)
