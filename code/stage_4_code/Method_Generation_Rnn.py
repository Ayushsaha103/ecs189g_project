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


class MethodGenerationRNN(method, nn.Module):
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
                 optimizer=optimizer, max_epoch=max_epoch, text_length=None, save_path=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # hyper params
        self.save_path = save_path
        self.save_dir = save_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.output_size = output_size
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.hidden_size = hidden_size
        self.max_epoch = max_epoch
        self.text_length = text_length
        self.vocab_size = vocab_size

        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        #rnn
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers=num_rnn_layers, batch_first=True, dropout=0.2)

        self.dropout = nn.Dropout(0.2)




        self.linear = nn.Linear(hidden_size, output_size*vocab_size)

        self.relu = nn.ReLU()

        # Metrics
        self.train_losses = []
        self.softplus = nn.Softplus()

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def to(self, device):
        self.device = device
        nn.Module.to(self, device)

    def forward(self, x, initial_hidden=None):
        x = self.embedding(x)
        x, hidden = self.rnn(x, initial_hidden)
        x = self.dropout(x)
        x = self.linear(x[:, -1])
        x = self.softplus(x)
        # x = self.relu(x)
        return x, hidden


    def generate(self, X, hidden=None):
        with torch.no_grad():
            return self.forward(torch.tensor(X, dtype=torch.int).to(self.device), hidden)
    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = self.loss_function(reduction='sum')
        # for training accuracy investigation purpose

        # Ensure metrics reset at start of training
        self.train_losses = []


        for epoch in range(self.max_epoch):
            epoch_loss = 0
            num_batches = np.ceil(X.shape[0] / self.batch_size)

            for batch_index in range(0, X.shape[0], self.batch_size):
                batch_X = X[batch_index:batch_index + self.batch_size]
                batch_y = y[batch_index:batch_index + self.batch_size]

                y_pred, hidden = self.forward(torch.tensor(batch_X, dtype=torch.int).to(self.device))
                y_true = torch.tensor(batch_y, dtype=torch.float).to(self.device)
                #
                # print(y_pred.shape, y_true.shape)
                # print(y_pred, y_true)


                # print(y_pred.shape, y_true.shape)
                train_loss = loss_function(y_pred, y_true)


                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                epoch_loss += train_loss.item()


            epoch_loss /= num_batches

            # Record loss for curr epoch/iteration
            self.train_losses.append(epoch_loss)


            if epoch % 1 == 0:
                print('Epoch:', epoch, 'Loss:', epoch_loss, end='\n')

        return self.train_losses[-1]

    def save_and_show_graph(self, fold_count: int):
        # After loop, plot recorded metrics
        fig = plt.figure(figsize=(10, 5))
        fig.suptitle(f'LR: {self.learning_rate} | Batch Size: {self.batch_size}'
                     f' | Loss: {self.loss_function} | Optimizer: {self.optimizer}', fontsize=16)

        plt.plot(self.train_losses, label="Training Loss", color='b')
        plt.title('Train Loss over time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Note: loss, accuracy plots generated here (nothing to be updated in search.py)
        self.folder_path = os.path.join(self.save_dir, 'graphs/')
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        plt.savefig(os.path.join(self.folder_path, f'fold-{fold_count}.png'))
        plt.show()



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
        loss = self.train(self.data['train']['X'], self.data['train']['y'])
        print('--saving graphs...')
        self.save_and_show_graph(fold_count)

        # free some memory
        self.data = None

        print('--saving model')
        torch.save(self, self.save_path)

        print('--returning')

        return {
            'loss': loss,
        }
