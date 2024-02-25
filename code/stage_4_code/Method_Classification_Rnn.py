'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_4_code.Evaulate_Classification_Metrics import Evaluate_Classification_Metrics
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import utils
import itertools


class Method_Classification_RNN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 50
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    batch_size = 1000
    save_dir = ''
    loss_function = nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    device = 'cpu'

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription, save_dir, vocab_size: int, embedding_size: int, hidden_size: int, num_rnn_layers:int,  output_size: int,
                 learning_rate=learning_rate, batch_size=batch_size, loss_function=loss_function,
                 optimizer=optimizer, max_epoch=max_epoch, text_length=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # hyper params
        self.save_dir = save_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.hidden_size = hidden_size
        self.metrics_evaluator = Evaluate_Classification_Metrics('evaluator', '')
        self.max_epoch = max_epoch
        self.text_length = text_length

        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        #rnn
        self.rnn = nn.RNN(embedding_size, hidden_size, num_layers=num_rnn_layers, dropout=0.35, batch_first=True, bidirectional=True)




        self.linear = nn.Linear(hidden_size*2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

        # Metrics
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
        self.test_data_accuracies = []
        self.test_data_precisions = []
        self.test_data_recalls = []
        self.test_data_f1s = []

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def to(self, device):
        self.device = device
        nn.Module.to(self, device)

    def forward(self, x):
        x = self.embedding(x)
        x, hidden = self.rnn(x)
        x = self.linear(x[:, -1])
        x = self.linear2(x)
        x = torch.squeeze(x)
        x = self.sigmoid(x)
        return x




    def train(self, X, y, test_data: None):
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
            num_batches = np.ceil(X.shape[0] / self.batch_size)

            for batch_index in range(0, X.shape[0], self.batch_size):
                batch_X = X[batch_index:batch_index + self.batch_size]
                batch_y = y[batch_index:batch_index + self.batch_size]

                y_pred = self.forward(torch.tensor(batch_X, dtype=torch.int).to(self.device))
                y_true = torch.tensor(batch_y, dtype=torch.float).to(self.device)


                # print(y_pred.shape, y_true.shape)
                train_loss = loss_function(y_pred, y_true)


                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                epoch_loss += train_loss.item()
                with torch.no_grad():

                    self.metrics_evaluator.data = {'true_y': y_true.cpu(), 'pred_y': y_pred.round().cpu()}
                    # print(y_true, y_pred)
                    curr_accuracy, precision, recall, f1 = self.metrics_evaluator.evaluate()
                    epoch_acc += curr_accuracy
                    epoch_precision += precision
                    epoch_recall += recall
                    epoch_f1 += f1


            epoch_loss /= num_batches
            epoch_acc /= num_batches
            epoch_precision /= num_batches
            epoch_recall /= num_batches
            epoch_f1 /= num_batches

            # Record loss for curr epoch/iteration
            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_acc)
            self.train_precisions.append(epoch_precision)
            self.train_recalls.append(epoch_recall)
            self.train_f1s.append(epoch_f1)

            # Record test file accuracy
            if test_data:
                with torch.no_grad():
                    test_batch_acc = 0
                    test_batch_precision = 0
                    test_batch_recall = 0
                    test_batch_f1 = 0
                    num_test_batches = X.shape[0] / self.batch_size + 1
                    y_pred = []
                    for batch_index in range(0, X.shape[0], self.batch_size):
                        test_data_y_pred = self.forward(
                            torch.tensor(X[batch_index:batch_index + self.batch_size], device=self.device)).cpu()
                        y_pred.extend(test_data_y_pred)
                        test_data_y_true = y[batch_index:batch_index + self.batch_size]
                        self.metrics_evaluator.data = {'true_y': test_data_y_true,
                                                       'pred_y': test_data_y_pred.round().cpu()}
                        accuracy, precision, recall, f1 = self.metrics_evaluator.evaluate()
                        test_batch_acc += accuracy
                        test_batch_precision += precision
                        test_batch_recall += recall
                        test_batch_f1 += f1

                    accuracy = test_batch_acc / num_test_batches
                    precision = test_batch_precision / num_test_batches
                    recall = test_batch_recall / num_test_batches
                    f1 = test_batch_f1 / num_test_batches

                    self.test_data_accuracies.append(accuracy)
                    self.test_data_f1s.append(f1)
                    self.test_data_recalls.append(recall)
                    self.test_data_precisions.append(precision)

            if epoch % 1 == 0:
                print('Epoch:', epoch, 'Accuracy:', epoch_acc, 'Loss:', epoch_loss, end=' | ')
                if test_data:
                    print('Test Data Acc:', self.test_data_accuracies[-1], 'f1:', self.test_data_f1s[-1], 'recall:',
                          self.test_data_recalls[-1], 'precision:', self.test_data_precisions[-1])

    def save_and_show_graph(self, fold_count: int):
        # After loop, plot recorded metrics
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(f'LR: {self.learning_rate} | Batch Size: {self.batch_size}'
                     f' | Loss: {self.loss_function} | Optimizer: {self.optimizer}', fontsize=16)

        plt.subplot(3, 3, 1)
        plt.plot(self.train_losses, label="Training Loss", color='b')
        plt.title('Train Loss over time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(3, 3, 2)
        plt.plot(self.train_accuracies, label="Training Accuracy", color='g')
        plt.title('Train Accuracy over time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(3, 3, 3)
        plt.plot(self.test_data_accuracies, label="Test File Accuracy", color='r')
        plt.title('Test File Accuracy over time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot for precision, recall, and F1 score
        plt.subplot(3, 3, 4)
        plt.plot(self.train_precisions, label="Training Precision", color='purple')
        plt.title('Train Precision over time')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()

        plt.subplot(3, 3, 5)
        plt.plot(self.train_recalls, label="Training Recall", color='orange')
        plt.title('Train Recall over time')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()

        plt.subplot(3, 3, 6)
        plt.plot(self.train_f1s, label="Training F1 Score", color='pink')
        plt.title('Train F1 Score over time')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()

        plt.subplot(3, 3, 7)
        plt.plot(self.test_data_precisions, label="Test File Precision", color='purple')
        plt.title('Test File Precision over time')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()

        plt.subplot(3, 3, 8)
        plt.plot(self.test_data_recalls, label="Test File Recall", color='orange')
        plt.title('Test File Recall over time')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()

        plt.subplot(3, 3, 9)
        plt.plot(self.test_data_f1s, label="Test File F1 Score", color='pink')
        plt.title('Test File F1 Score over time')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()

        # Adjust params so subplots fit in figure area
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))

        # Note: loss, accuracy plots generated here (nothing to be updated in search.py)
        self.folder_path = os.path.join(self.save_dir, 'graphs/')
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        plt.savefig(os.path.join(self.folder_path, f'fold-{fold_count}.png'))
        plt.show()


    @torch.no_grad()
    def test(self, X, y):
        test_batch_acc = 0
        test_batch_precision = 0
        test_batch_recall = 0
        test_batch_f1 = 0
        num_test_batches = X.shape[0] / self.batch_size + 1
        y_pred = []
        for batch_index in range(0, X.shape[0], self.batch_size):
            test_data_y_pred = self.forward(
                torch.tensor(X[batch_index:batch_index + self.batch_size], device=self.device)).cpu()
            y_pred.extend(test_data_y_pred)
            test_data_y_true = y[batch_index:batch_index + self.batch_size]
            self.metrics_evaluator.data = {'true_y': test_data_y_true, 'pred_y': test_data_y_pred.round().cpu()}
            accuracy, precision, recall, f1 = self.metrics_evaluator.evaluate()
            test_batch_acc += accuracy
            test_batch_precision += precision
            test_batch_recall += recall
            test_batch_f1 += f1

        accuracy = test_batch_acc / num_test_batches
        precision = test_batch_precision / num_test_batches
        recall = test_batch_recall / num_test_batches
        f1 = test_batch_f1 / num_test_batches
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


    def run(self, fold_count: int):
        print('method running...')

        self.reset()

        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'], self.data['test_data'])
        print('--saving graphs...')
        self.save_and_show_graph(fold_count)
        print('--start testing...')
        # pred_y = self.test(self.data['test']['X'])
        pred_y, accuracy, precision, recall, f1 = self.test(self.data['test_data']['X'], self.data['test_data']['y'])

        return {
            'pred_y': pred_y,
            'true_y': self.data['test_data']['y'],
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
