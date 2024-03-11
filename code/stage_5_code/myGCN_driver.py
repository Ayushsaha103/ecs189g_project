


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# citation:
# credits to Thomas N. and Welling for the GCN model code:
# @article{kipf2016semi,
#   title={Semi-Supervised Classification with Graph Convolutional Networks},
#   author={Kipf, Thomas N and Welling, Max},
#   journal={arXiv preprint arXiv:1609.02907},
#   year={2016}
# }



from code.stage_5_code.pygcn.utils import load_data, accuracy
from code.stage_5_code.pygcn.models import GCN
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from code.stage_5_code.GCN_cora import  GCN_cora
from code.stage_5_code.GCN_pubmed import GCN_pubmed
from code.stage_5_code.GCN_citeseer import GCN_citeseer



class GCN_driver():
    def __init__(self, epochs_=200, lr_=0.01, wt_decay=0.0005, nhidden_=16, dropout_=0.5):
        # Training settings
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--no-cuda', action='store_true', default=False,
                                 help='Disables CUDA training.')
        self.parser.add_argument('--fastmode', action='store_true', default=False,
                                 help='Validate during training pass.')
        self.parser.add_argument('--seed', type=int, default=42, help='Random seed.')
        self.parser.add_argument('--epochs', type=int, default=epochs_,
                                 help='Number of epochs to train.')
        self.parser.add_argument('--lr', type=float, default=lr_,
                                 help='Initial learning rate.')
        self.parser.add_argument('--weight_decay', type=float, default=wt_decay,
                                 help='Weight decay (L2 loss on parameters).')
        self.parser.add_argument('--hidden', type=int, default=nhidden_,
                                 help='Number of hidden units.')
        self.parser.add_argument('--dropout', type=float, default=dropout_,
                                 help='Dropout rate (1 - keep probability).')

        self.args = self.parser.parse_args()
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()

        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.args.cuda:
            torch.cuda.manual_seed(self.args.seed)

    def load_data(self, D, dataset_name):

        #
        # # Load data
        # adj, features, labels, idx_train, idx_val, idx_test = load_data()

        self.adj = D['graph']['utility']['A']
        self.features = D['graph']['X']
        self.labels = D['graph']['y']
        self.idx_train = D['train_test_val']['idx_train']
        # idx_val = D['train_test_val']['idx_val']
        self.idx_test = D['train_test_val']['idx_test']

        # Model and optimizer
        # self.model = GCN(nfeat=self.features.shape[1],
        #             nhid=self.args.hidden,
        #             nclass=self.labels.max().item() + 1,
        #             dropout=self.args.dropout)
        self.model = None
        if dataset_name == 'cora':
            self.model = GCN_cora(nfeat=self.features.shape[1],
                             nhid=self.args.hidden,
                             nclass=self.labels.max().item() + 1,
                             dropout=self.args.dropout)
        elif dataset_name == 'pubmed':
            self.model = GCN_pubmed(nfeat=self.features.shape[1],
                             nhid=self.args.hidden,
                             nclass=self.labels.max().item() + 1,
                             dropout=self.args.dropout)
        elif dataset_name == 'citeseer':
            self.model = GCN_citeseer(nfeat=self.features.shape[1],
                             nhid=self.args.hidden,
                             nclass=self.labels.max().item() + 1,
                             dropout=self.args.dropout)

        self.optimizer = optim.Adam(self.model.parameters(),
                               lr=self.args.lr, weight_decay=self.args.weight_decay)

        if self.args.cuda:
            self.model.cuda()
            self.features = self.features.cuda()
            self.adj = self.adj.cuda()
            self.labels = self.labels.cuda()
            self.idx_train = self.idx_train.cuda()
            # idx_val = idx_val.cuda()
            self.idx_test = self.idx_test.cuda()


    def train_single_epoch(self, epoch):
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(self.features, self.adj)
        loss_train = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])
        acc_train = accuracy(output[self.idx_train], self.labels[self.idx_train])
        loss_train.backward()
        self.optimizer.step()

        if not self.args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            self.model.eval()
            output = self.model(self.features, self.adj)

        # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        # acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch +1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              # 'loss_val: {:.4f}'.format(loss_val.item()),
              # 'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t), end = "\t\t")
        return loss_train.item(), acc_train.item()

    def test(self):
        self.model.eval()
        output = self.model(self.features, self.adj)
        loss_test = F.nll_loss(output[self.idx_test], self.labels[self.idx_test])
        acc_test = accuracy(output[self.idx_test], self.labels[self.idx_test])
        # print(output[self.idx_test])
        # print(self.labels[self.idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return loss_test.item(), acc_test.item()


    def plot_learning_curves(self, train_losses, train_accs, test_losses, test_accs):
        plt.figure(figsize=(12, 6))
        epochs = range(self.args.epochs)

        # Plot accuracy curves
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_accs, label='Training Accuracy', marker='o')
        plt.plot(epochs, test_accs, label='Test Accuracy', marker='o')
        plt.title('Accuracy Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        # Plot loss curves
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_losses, label='Training Loss', marker='o')
        plt.plot(epochs, test_losses, label='Test Loss', marker='o')
        plt.title('Loss Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Adjust layout
        plt.tight_layout()

        # Show plot
        plt.show()

    def train(self):

        # Train model
        t_total = time.time()
        train_losses, train_accs = [], []
        test_losses, test_accs = [], []

        for epoch in range(self.args.epochs):
            train_loss, train_acc = self.train_single_epoch(epoch)
            test_loss, test_acc = self.test()

            train_accs.append(train_acc)
            train_losses.append(train_loss)
            test_accs.append(test_acc)
            test_losses.append(test_loss)

        # import pdb; pdb.set_trace()
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        self.plot_learning_curves(train_losses, train_accs, test_losses, test_accs)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# citation:
# credits to Thomas N. and Welling for the GCN model code:
# @article{kipf2016semi,
#   title={Semi-Supervised Classification with Graph Convolutional Networks},
#   author={Kipf, Thomas N and Welling, Max},
#   journal={arXiv preprint arXiv:1609.02907},
#   year={2016}
# }