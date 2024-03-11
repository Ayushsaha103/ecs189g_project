from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import precision_recall_fscore_support
#from torch.utils.data import DataLoader
from code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from pygcn.models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
# parser.add_argument('--fastmode', action='store_true', default=False,
#                     help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
#adj, features, labels, idx_train, idx_val, idx_test = load_data()
dataset_loader = Dataset_Loader(dName='cora', dDescription='Cora Dataset')
data = dataset_loader.load()
adj = data['graph']['utility']['A']
features = data['graph']['X']
labels = data['graph']['y']
idx_train = data['train_test_val']['idx_train']
idx_val = data['train_test_val']['idx_val']
idx_test = data['train_test_val']['idx_test']

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    #acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # if not args.fastmode:
    #     # Evaluate validation set performance separately,
    #     # deactivates dropout during validation run.
    #     model.eval()
    #     output = model(features, adj)

    if args.cuda:
        torch.cuda.synchronize()

    # Evaluate metrics
    model.eval()
    output = model(features, adj)
    predicts = output.max(1)[1].type_as(labels)
    precision, recall, f1, _ = precision_recall_fscore_support(labels[idx_train].cpu(), predicts[idx_train].cpu(),
                                                               average='micro')
    acc_train = predicts[idx_train].eq(labels[idx_train]).double().sum() / len(idx_train)


    # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    # acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'accuracy_train: {:.4f}'.format(acc_train.item()),
          'precision_train: {:.4f}'.format(precision),
          'recall_train: {:.4f}'.format(recall),
          'f1_train: {:.4f}'.format(f1),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)

    #loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    #acc_test = accuracy(output[idx_test], labels[idx_test])

    preds = output.max(1)[1].type_as(labels)
    precision, recall, f1, _ = precision_recall_fscore_support(labels[idx_test].cpu(), preds[idx_test].cpu(), average='micro')
    acc_test = preds[idx_test].eq(labels[idx_test]).double().sum() / len(idx_test)


    print("Test set results:",
          "loss= {:.4f}".format(F.nll_loss(output[idx_test], labels[idx_test]).item()),
          "accuracy= {:.4f}".format(acc_test.item()),
          "precision= {:.4f}".format(precision),
          "recall= {:.4f}".format(recall),
          "f1= {:.4f}".format(f1))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
