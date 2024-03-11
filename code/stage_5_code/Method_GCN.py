'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
import torch

import matplotlib.pyplot as plt
import os

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 50
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-2
    save_dir = ''
    loss_function = nn.NLLLoss
    optimizer = torch.optim.Adam
    device = 'cpu'


    def __init__(self, mName, mDescription, dir, nfeat, nhid, nclass, dropout):

        super(GCN, self).__init__()
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.save_dir = dir
        self.metrics_evaluator = Evaluate_Metrics('evaluator', '')

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
        x = self.conv3(x, adj)
        return F.log_softmax(x, dim=1)

    def train(self, feature, adj, label, indx, test_data=None):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = self.loss_function()
        # for training accuracy investigation purpose

        # Ensure metrics reset at start of training
        self.train_losses = []
        self.train_accuracies = []
        self.train_precisions = []
        self.train_recalls = []
        self.train_f1s = []
        self.val_losses = []
        self.val_accuracies = []
        self.precision_scores = []
        self.recall_scores = []
        self.f1_scores = []
        self.test_data_losses = []
        self.test_data_accuracies = []
        self.test_data_precisions = []
        self.test_data_recalls = []
        self.test_data_f1s = []

        for epoch in range(self.max_epoch):
            epoch_loss = 0
            epoch_acc = 0
            epoch_precision = 0
            epoch_recall = 0
            epoch_f1 = 0
            # num_batches = np.ceil(X.shape[0] / self.batch_size)

            y_pred = self.forward(feature, adj)
            y_true = label

            train_loss = loss_function(y_pred[indx], y_true[indx])

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            epoch_loss += train_loss.item()
            with torch.no_grad():
                self.metrics_evaluator.data = {'true_y': y_true[indx].cpu(), 'pred_y': y_pred[indx].max(1)[1].cpu()}
                curr_accuracy, precision, recall, f1 = self.metrics_evaluator.evaluate()
                epoch_acc += curr_accuracy
                epoch_precision += precision
                epoch_recall += recall
                epoch_f1 += f1



            # Record loss for curr epoch/iteration
            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_acc)
            self.train_precisions.append(epoch_precision)
            self.train_recalls.append(epoch_recall)
            self.train_f1s.append(epoch_f1)

            # Record test file accuracy
            if test_data:
                with torch.no_grad():
                    test_data_y_pred = self.forward(test_data['feature'], test_data['adj']).cpu()
                    test_data_y_true = test_data['label'].cpu()
                    test_loss = loss_function(test_data_y_pred[indx], test_data_y_true[indx])
                    self.metrics_evaluator.data = {'true_y': test_data_y_true[test_data['indx']], 'pred_y': test_data_y_pred[test_data['indx']].max(1)[1]}
                    accuracy, precision, recall, f1 = self.metrics_evaluator.evaluate()
                    self.test_data_losses.append(test_loss)
                    self.test_data_accuracies.append(accuracy)
                    self.test_data_precisions.append(precision)
                    self.test_data_recalls.append(recall)
                    self.test_data_f1s.append(f1)

            if epoch % 5 == 0:
                if test_data:
                    print(f"Epoch: {epoch} | Accuracy: {epoch_acc:.2f} | Loss: {epoch_loss:.2f} | "
                          f"Test Loss: {self.test_data_losses[-1]:.2f} | "
                          f"Test Data Acc: {self.test_data_accuracies[-1]:.2f} | "
                          f"f1: {self.test_data_f1s[-1]:.2f} | "
                          f"recall: | {self.test_data_recalls[-1]:.2f} | "
                          f"precision: {self.test_data_precisions[-1]:.2f}")

    def save_and_show_graph(self, fold_count: int):
        # After loop, plot recorded metrics
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(f'LR: {self.learning_rate}'
                     f' | Loss: {self.loss_function} | Optimizer: {self.optimizer}', fontsize=16)

        # Loss
        plt.subplot(2, 3, 1)
        plt.title('Loss over time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.plot(self.train_losses, label="Training Loss", color='g')
        plt.plot(self.test_data_losses, label="Testing Loss", color='r')
        plt.legend()

        # Accuracy
        plt.subplot(2, 3, 2)
        plt.title('Accuracy over time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.plot(self.train_accuracies, label="Training Accuracy", color='g')
        plt.plot(self.test_data_accuracies, label="Testing Accuracy", color='r')
        plt.legend()

        # Precision
        plt.subplot(2, 3, 3)
        plt.title('Precision over time')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')

        plt.plot(self.train_precisions, label="Training Precision", color='g')
        plt.plot(self.test_data_precisions, label="Testing Precision", color='r')
        plt.legend()

        # Recall
        plt.subplot(2, 3, 4)
        plt.title('Recall over time')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')

        plt.plot(self.train_recalls, label="Training Recall", color='g')
        plt.plot(self.test_data_recalls, label="Testing Recall", color='r')
        plt.legend()

        # F1
        plt.subplot(2, 3, 5)
        plt.title('F1 Score over time')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')

        plt.plot(self.train_f1s, label="Training F1 Score", color='g')
        plt.plot(self.test_data_f1s, label="TestingF1 Score", color='r')
        plt.legend()

        # Adjust params so subplots fit in figure area
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        # Note: loss, accuracy plots generated here (nothing to be updated in search.py)
        self.folder_path = os.path.join(self.save_dir, 'graphs/')
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        plt.savefig(os.path.join(self.folder_path, f'{fold_count}.png'))
        plt.show()

    @torch.no_grad()
    def test(self, feature, adj, label, indx):
        y_pred = self.forward(feature, adj).cpu()
        y_true = label.cpu()
        self.metrics_evaluator.data = {'true_y': y_true[indx], 'pred_y': y_pred[indx].max(1)[1]}
        accuracy, precision, recall, f1 = self.metrics_evaluator.evaluate()
        return y_pred, accuracy, precision, recall, f1

    def reset(self):
        @torch.no_grad()
        def weight_reset(m: nn.Module):
            # - check if the current module has reset_parameters & if it's callabed called it on m
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

        self.apply(fn=weight_reset)
        self.train_accuracies = []
        self.train_losses = []
        self.test_data_losses = []


    def run(self):
        print('method running...')

        self.reset()

        print('--start training...')
        self.train(self.data['train']['feature'], self.data['train']['adj'],  self.data['train']['label'], self.data['train']['indx'], self.data['test'])
        print('--saving graphs...')
        self.save_and_show_graph(1)
        print('--start testing...')
        pred_y, accuracy, precision, recall, f1 = self.test(self.data['test']['feature'], self.data['test']['adj'],  self.data['test']['label'], self.data['test']['indx'])

        return {
            'pred_y': pred_y,
            'true_y': self.data['test']['label'],
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
