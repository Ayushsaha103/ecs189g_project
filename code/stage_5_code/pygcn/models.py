import torch.nn as nn
import torch.nn.functional as F
# from pygcn.layers import GraphConvolution
from torch_geometric.nn import GCNConv
import time

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        self.conv3 = GCNConv(nhid, nhid)
        self.conv4 = GCNConv(nhid, nhid)
        self.conv5 = GCNConv(nhid, nhid)
        self.conv6 = GCNConv(nhid, nhid)
        self.conv7 = GCNConv(nhid, nhid)
        self.conv8 = GCNConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv4(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv5(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv6(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv7(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv8(x, adj)
        return F.log_softmax(x, dim=1)

    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        # output = model(features)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(features, adj)

        # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        # acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              # 'loss_val: {:.4f}'.format(loss_val.item()),
              # 'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

