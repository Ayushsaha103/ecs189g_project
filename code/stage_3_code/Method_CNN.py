'''
Concrete MethodModule class for a specific learning MethodModule
'''
import torchvision

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import utils
import itertools


class Method_CNN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 50
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    batch_size = 1000
    save_dir = ''
    loss_function = nn.CrossEntropyLoss
    optimizer = torch.optim.Adam

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription, save_dir, input_shape: int, hidden_units: int, output_shape: int,
                 learning_rate=learning_rate, batch_size=batch_size, loss_function=loss_function,
                 optimizer=optimizer, max_epoch=max_epoch, output_layer_input_channels=5):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # hyper params
        self.save_dir = save_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.metrics_evaluator = Evaluate_Metrics('evaluator', '')
        self.max_epoch = max_epoch
        self.output_layer_input_channels = output_layer_input_channels


        # first conv is seperate so we can visualize the kernels later
        self.first_conv = nn.Conv2d(in_channels=input_shape,
                                    out_channels=hidden_units,
                                    kernel_size=8,
                                    stride=1,
                                    padding=0)

        self.conv_block_1 = nn.Sequential(
            self.first_conv,
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=5,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,
                         stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=output_layer_input_channels,
                      out_features=output_shape)
        )

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

    def forward(self, x):
        c1 = self.conv_block_1(x)
        c2 = self.conv_block_2(c1)
        y_pred = self.classifier(c2)
        return y_pred

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
        self.val_losses = []
        self.val_accuracies = []
        self.precision_scores = []
        self.recall_scores = []
        self.f1_scores = []
        self.test_data_accuracies = []
        self.test_data_precisions = []
        self.test_data_recalls = []
        self.test_data_f1s = []

        for epoch in range(self.max_epoch):
            epoch_loss = 0
            epoch_acc = 0
            num_batches = np.ceil(X.shape[0] / self.batch_size)

            for batch_index in range(0, X.shape[0], self.batch_size):
                batch_X = X[batch_index:batch_index + self.batch_size]
                batch_y = y[batch_index:batch_index + self.batch_size]

                y_pred = self.forward(torch.FloatTensor(np.array(batch_X)))
                y_true = torch.LongTensor(np.array(batch_y))

                if self.loss_function == nn.CrossEntropyLoss:
                    train_loss = loss_function(y_pred, y_true)
                else:
                    train_loss = loss_function(y_pred,
                                               nn.functional.one_hot(y_true, num_classes=10).type(torch.FloatTensor))

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                epoch_loss += train_loss.item()
                with torch.no_grad():
                    self.metrics_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                    curr_accuracy, precision, recall, f1 = self.metrics_evaluator.evaluate()
                    epoch_acc += curr_accuracy
                    self.train_precisions.append(precision)
                    self.train_recalls.append(recall)
                    self.train_f1s.append(f1)

            epoch_loss /= num_batches
            epoch_acc /= num_batches

            # Record loss for curr epoch/iteration
            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_acc)

            # Record test file accuracy
            if test_data:
                with torch.no_grad():
                    test_data_y_pred = self.forward(torch.FloatTensor(np.array(test_data['X'])))
                    test_data_y_true = torch.LongTensor(np.array(test_data['y']))
                    self.metrics_evaluator.data = {'true_y': test_data_y_true, 'pred_y': test_data_y_pred.max(1)[1]}
                    accuracy, precision, recall, f1 = self.metrics_evaluator.evaluate()
                    self.test_data_accuracies.append(accuracy)
                    self.test_data_precisions.append(precision)
                    self.test_data_recalls.append(recall)
                    self.test_data_f1s.append(f1)

            if epoch % 5 == 0:
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
        plt.plot(self.train_precisions, label="Test File Precision", color='purple')
        plt.title('Test File Precision over time')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()

        plt.subplot(3, 3, 8)
        plt.plot(self.train_recalls, label="Test File Recall", color='orange')
        plt.title('Test File Recall over time')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()

        plt.subplot(3, 3, 9)
        plt.plot(self.train_f1s, label="Test File F1 Score", color='pink')
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

    def plot_first_conv_kernels(self, fold_count):
        kernels = self.first_conv.weight.data.detach().clone()
        kernels = kernels - kernels.min()
        kernels = kernels / kernels.max()
        filter_img = torchvision.utils.make_grid(kernels, nrow=10).permute(1, 2, 0).numpy()
        plt.imshow(filter_img)
        plt.imsave(os.path.join(self.folder_path, f'fold-{fold_count}-first-layer-kernels.png'), filter_img)

    @torch.no_grad()
    def test(self, X, y):
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        y_true = torch.LongTensor(np.array(y))
        self.metrics_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
        accuracy, precision, recall, f1 = self.metrics_evaluator.evaluate()
        return y_pred, accuracy, precision, recall, f1

    def run(self, fold_count: int):
        print('method running...')

        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'], self.data['test_data'])
        print('--saving graphs...')
        self.save_and_show_graph(fold_count)
        print('--start testing...')
        # pred_y = self.test(self.data['test']['X'])
        pred_y, accuracy, precision, recall, f1 = self.test(self.data['test']['X'], self.data['test']['y'])

        self.plot_first_conv_kernels(fold_count)

        confusion_matrix = self.metrics_evaluator.generate_confusion_matrix()
        confusion_matrix.plot()
        plt.savefig(os.path.join(self.folder_path, f'fold-{fold_count}-cm.png'))
        plt.show()
        return {
            'pred_y': pred_y,
            'true_y': self.data['test']['y'],
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
