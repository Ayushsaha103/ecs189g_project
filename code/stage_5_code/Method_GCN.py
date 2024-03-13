import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from code.base_class.method import method
from .pygcn.layers import GraphConvolution
from .Evaluate_Metrics import Evaluate_Metrics
import matplotlib.pyplot as plt
from code.stage_5_code.pygcn.utils import accuracy as compute_accuracy
import argparse

class Method_GCN(method, nn.Module):
    # Base class for GCN methods
    def __init__(self, mName, mDescription, nfeat, nhid, nclass, epochs, dropout_rate, layers, learning_rate, weight_decay):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate

        # Define GCN layers
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvolution(nfeat, nhid)) # First layer

        # If more than 2 layers, add middle layers
        for i in range(1, layers - 1):
            self.layers.append(GraphConvolution(nhid, nhid))
        self.layers.append(GraphConvolution(nhid, nclass))

        # Define metrics evaluator
        self.metrics_evaluator = Evaluate_Metrics('evaluator', '')

        # Training settings
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--no-cuda', action='store_true', default=False,
                                 help='Disables CUDA training.')
        self.parser.add_argument('--fastmode', action='store_true', default=False,
                                 help='Validate during training pass.')
        self.parser.add_argument('--seed', type=int, default=42, help='Random seed.')
        self.parser.add_argument('--epochs', type=int, default=epochs,
                                 help='Number of epochs to train.')
        self.parser.add_argument('--lr', type=float, default=learning_rate,
                                 help='Initial learning rate.')
        self.parser.add_argument('--weight_decay', type=float, default=weight_decay,
                                 help='Weight decay (L2 loss on parameters).')
        self.parser.add_argument('--hidden', type=int, default=nhid,
                                 help='Number of hidden units.')
        self.parser.add_argument('--dropout', type=float, default=dropout_rate,
                                 help='Dropout rate (1 - keep probability).')

        self.args = self.parser.parse_args()
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()

        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.args.cuda:
            torch.cuda.manual_seed(self.args.seed)

        # Initialize the cross entropy loss
        self.criterion = torch.nn.CrossEntropyLoss()

        # Initialize lists to store loss, accuracy for plotting
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.test_precisions = []
        self.test_recalls = []
        self.test_f1_scores = []

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout_rate, training=self.training)
        return x  # Removed log_softmax

    def train_single_epoch(self, epoch):
        t = time.time()
        self.train()  # Use self to reference the model for training
        self.optimizer.zero_grad()
        output = self(self.features, self.adj)  # Call forward pass
        # Update loss calculation to use cross entropy loss
        loss_train = self.criterion(output[self.idx_train], self.labels[self.idx_train])
        acc_train = compute_accuracy(output[self.idx_train], self.labels[self.idx_train])
        loss_train.backward()
        self.optimizer.step()

        if not self.args.fastmode:
            self.eval()  # Evaluation mode
            output = self(self.features, self.adj)  # Forward pass for evaluation

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'time: {:.4f}s'.format(time.time() - t), end="\t\t")
        return loss_train.item(), acc_train.item()

    def train_model(self):
        t_total = time.time()

        for epoch in range(self.args.epochs):
            train_loss, train_acc = self.train_single_epoch(epoch)
            test_loss, test_acc, precision, recall, f1 = self.test()

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_acc)
            self.test_precisions.append(precision)
            self.test_recalls.append(recall)
            self.test_f1_scores.append(f1)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    @torch.no_grad()
    def test(self):
        self.eval()  # Ensure the model is in evaluation mode
        output = self(self.features, self.adj)  # Get model predictions for the test data
        # Update loss calculation to use cross entropy loss
        loss_test = self.criterion(output[self.idx_test], self.labels[self.idx_test])
        acc_test = compute_accuracy(output[self.idx_test], self.labels[self.idx_test])

        # Convert predictions to class labels
        preds = output[self.idx_test].max(1)[1]

        # Ensure labels, predictions are on CPU & converted to NumPy for sklearn metrics
        true_labels = self.labels[self.idx_test].cpu().numpy()
        pred_labels = preds.cpu().numpy()

        # Set the data for metrics evaluation
        self.metrics_evaluator.data = {
            'true_y': true_labels,
            'pred_y': pred_labels
        }
        # Use metrics evaluator to calculate precision, recall, and f1
        accuracy, precision, recall, f1 = self.metrics_evaluator.evaluate()

        print("Test set results:",
              f"loss= {loss_test.item():.4f}",
              f"accuracy= {acc_test.item():.4f}",
              f"precision= {precision:.4f}",
              f"recall= {recall:.4f}",
              f"F1 score= {f1:.4f}")
        return loss_test.item(), acc_test.item(), precision, recall, f1

    @torch.no_grad()
    def evaluate(self, output, labels, mask):
        preds = output.max(1)[1].type_as(labels)
        correct = preds[mask].eq(labels[mask]).double()
        correct = correct.sum()
        return preds, correct / mask.sum().item()

    def plot_metrics(self, dataset_name, learning_rate):
        # Plotting
        fig, axs = plt.subplots(1, 2, figsize=(20, 8))  # Keep the figsize to make plots larger
        fig.suptitle(f'Dataset: {dataset_name} | LR: {learning_rate} | Loss: Cross-Entropy | Optimizer: Adam', fontsize=16)

        # Plot training loss and accuracy
        axs[0].plot(self.train_losses, label='Train Loss')
        axs[0].plot(self.train_accuracies, label='Train Accuracy')
        axs[0].set_title('Training Loss & Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].legend()

        # Plot testing metrics
        axs[1].plot(self.test_losses, label='Test Loss')
        axs[1].plot(self.test_accuracies, label='Test Accuracy')
        axs[1].plot(self.test_precisions, label='Test Precision')
        axs[1].plot(self.test_recalls, label='Test Recall')
        axs[1].plot(self.test_f1_scores, label='Test F1 Score')
        axs[1].set_title('Testing Metrics')
        axs[1].set_xlabel('Epoch')
        axs[1].legend()

        # Adjust the layout to make room for the suptitle and prevent overlap
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # Reduce horizontal space between the plots
        plt.subplots_adjust(wspace=0.1)  # Reduced wspace value

        plt.show()
