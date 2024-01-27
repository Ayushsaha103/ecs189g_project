'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools


class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 100
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    batch_size = 1000
    save_dir = ''
    loss_function = nn.CrossEntropyLoss
    optimizer = torch.optim.Adam

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription, save_dir, learning_rate=learning_rate, batch_size=batch_size, loss_function=loss_function, optimizer=optimizer):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # hyper params
        self.save_dir = save_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.optimizer = optimizer

        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc_layer_1 = nn.Linear(784, 4)
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.activation_func_1 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(4, 10)
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        self.activation_func_2 = nn.Softmax(dim=1)

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
        self.test_file_accuracies = []

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        h = self.activation_func_1(self.fc_layer_1(x))
        # outout layer result
        # self.fc_layer_2(h) will be a nx2 tensor
        # n (denotes the input instance number): 0th dimension; 2 (denotes the class number): 1st dimension
        # we do softmax along dim=1 to get the normalized classification probability distributions for each instance
        y_pred = self.activation_func_2(self.fc_layer_2(h))
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    @staticmethod
    def evaluate_metrics(true_labels, predicted_labels, average="weighted"):
        precision = precision_score(true_labels, predicted_labels, average=average, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, average=average, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, average=average, zero_division=0)
        return precision, recall, f1

    def train(self, X, y, test_file_data: None):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = self.loss_function()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # Ensure metrics reset at start of training
        self.train_losses = []
        self.train_accuracies = []

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...
            epoch_loss = 0
            epoch_acc = 0
            num_batches = np.ceil(X.shape[0] / self.batch_size)

            for batch_index in range(0, X.shape[0], self.batch_size):
                batch_X = X[batch_index:batch_index + self.batch_size]
                batch_y = y[batch_index:batch_index + self.batch_size]

                # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
                y_pred = self.forward(torch.FloatTensor(np.array(batch_X)))
                # convert y to torch.tensor as well
                y_true = torch.LongTensor(np.array(batch_y))
                # calculate the training loss

                if self.loss_function == nn.CrossEntropyLoss:
                    train_loss = loss_function(y_pred, y_true)
                else:
                    train_loss = loss_function(y_pred, nn.functional.one_hot(y_true, num_classes=10).type(torch.FloatTensor))

                # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
                optimizer.zero_grad()
                # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
                # do the error backpropagation to calculate the gradients
                train_loss.backward()
                # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
                # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
                optimizer.step()
                epoch_loss += train_loss.item()
                with torch.no_grad():
                    accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                    curr_accuracy = accuracy_evaluator.evaluate()
                    epoch_acc += curr_accuracy
                    precision, recall, f1 = self.evaluate_metrics(y_true, y_pred.max(1)[1])
                    self.train_precisions.append(precision)
                    self.train_recalls.append(recall)
                    self.train_f1s.append(f1)

            epoch_loss /= num_batches
            epoch_acc /= num_batches

            # Record loss for curr epoch/iteration
            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_acc)

            # Record test file accuracy
            if test_file_data:
                with torch.no_grad():
                    test_file_y_pred = self.forward(torch.FloatTensor(np.array(test_file_data['X'])))
                    test_file_y_true = torch.LongTensor(np.array(test_file_data['y']))
                    accuracy_evaluator.data = {'true_y': test_file_y_true, 'pred_y': test_file_y_pred.max(1)[1]}
                    self.test_file_accuracies.append(accuracy_evaluator.evaluate())

            if epoch % 20 == 0:
                print('Epoch:', epoch, 'Accuracy:', epoch_acc, 'Loss:', epoch_loss, end=' | ')
                if test_file_data:
                    print('Test File Acc:', self.test_file_accuracies[-1])

    def save_and_show_graph(self, fold_count: int):
        # After loop, plot recorded metrics
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(f'LR: {self.learning_rate} | Batch Size: {self.batch_size}'
                     f' | Loss: {self.loss_function} | Optimizer: {self.optimizer}', fontsize=16)

        plt.subplot(2, 3, 1)
        plt.plot(self.train_losses, label="Training Loss", color='b')
        plt.title('Train Loss over time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 3, 2)
        plt.plot(self.train_accuracies, label="Training Accuracy", color='g')
        plt.title('Train Accuracy over time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(2, 3, 3)
        plt.plot(self.test_file_accuracies, label="Test File Accuracy", color='r')
        plt.title('Test File Accuracy over time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot for precision, recall, and F1 score
        plt.subplot(2, 3, 4)
        plt.plot(self.train_precisions, label="Training Precision", color='purple')
        plt.title('Train Precision over time')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()

        plt.subplot(2, 3, 5)
        plt.plot(self.train_recalls, label="Training Recall", color='orange')
        plt.title('Train Recall over time')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()

        plt.subplot(2, 3, 6)
        plt.plot(self.train_f1s, label="Training F1 Score", color='pink')
        plt.title('Train F1 Score over time')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()

        # Adjust params so subplots fit in figure area
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))

        # Note: loss, accuracy plots generated here (nothing to be updated in main.py)
        folder_path = os.path.join(self.save_dir, 'graphs/')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(os.path.join(folder_path, f'fold-{fold_count}.png'))
        plt.show()

    def test(self, X, y):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
       # return y_pred.max(1)[1]

        # y_true = torch.FloatTensor(np.array(y))
        y_true = torch.LongTensor(np.array(y))  # This should be LongTensor for classification targets

        # Get numpy arrays from tensors for evaluation
        y_true_np = y_true.numpy()
        y_pred_np = y_pred.max(1)[1].numpy()
        precision, recall, f1 = self.evaluate_metrics(y_true_np, y_pred_np)
        return y_pred.max(1)[1], precision, recall, f1

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
        self.train(self.data['train']['X'], self.data['train']['y'], self.data['test_file'])
        print('--saving graphs...')
        self.save_and_show_graph(fold_count)
        print('--start testing...')
        # pred_y = self.test(self.data['test']['X'])
        pred_y, precision, recall, f1 = self.test(self.data['test']['X'], self.data['test']['y'])

        # return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
        return {
            'pred_y': pred_y,
            'true_y': self.data['test']['y'],
            'precision': precision,
            'recall': recall,
            'f1': f1
        }