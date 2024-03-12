import torch
import torch.nn as nn
import torch.nn.functional as F
from .pygcn.layers import GraphConvolution
from code.base_class.method import method

import numpy as np
np.random.seed(42)
torch.manual_seed(42)
class GCN_cora(method, nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout_rate, layers):
        super(GCN_cora, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(GraphConvolution(nfeat, nhid))  # First layer

        # If more than 2 layers, add middle layers
        for i in range(1, layers - 1):
            self.layers.append(GraphConvolution(nhid, nhid))

        self.layers.append(GraphConvolution(nhid, nclass))  # Last layer

        # Dropout rate
        self.dropout_rate = dropout_rate

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout_rate, training=self.training)
        return F.log_softmax(x, dim=1)