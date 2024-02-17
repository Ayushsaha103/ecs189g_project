'''
Concrete MethodModule class for a specific learning MethodModule
'''
import torchvision

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
from code.stage_3_code.Method_CNN import Method_CNN
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import utils
import itertools

class Method_CNN_CIFAR10(Method_CNN, method, nn.Module):
    def __init__(self, mName, mDescription, save_dir, input_shape: int, first_conv_hidden_units: int, second_conv_hidden_units: int, third_conv_hidden_units: int, output_shape: int,
                 learning_rate, batch_size, loss_function,
                 optimizer, max_epoch, output_layer_input_channels):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        Method_CNN.__init__(self, mName, mDescription, save_dir, input_shape, first_conv_hidden_units, output_shape, learning_rate,
                            batch_size, loss_function, optimizer, max_epoch, output_layer_input_channels)

        # first conv is seperate so we can visualize the kernels later
        self.first_conv = nn.Conv2d(in_channels=input_shape,
                                    out_channels=first_conv_hidden_units,
                                    kernel_size=3,
                                    stride=2,
                                    padding=2)

        self.conv_block_1 = nn.Sequential(
            self.first_conv,
            nn.ReLU(),
            nn.Conv2d(in_channels=first_conv_hidden_units,
                      out_channels=first_conv_hidden_units,
                      kernel_size=3,
                      stride=2,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(first_conv_hidden_units, second_conv_hidden_units, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.Conv2d(second_conv_hidden_units, second_conv_hidden_units, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(second_conv_hidden_units, third_conv_hidden_units, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.Conv2d(third_conv_hidden_units, third_conv_hidden_units, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=output_layer_input_channels,
                      out_features=output_shape)
        )

    def forward(self, x):
        c1 = self.conv_block_1(x)
        c2 = self.conv_block_2(c1)
        c3 = self.conv_block_3(c2)
        y_pred = self.classifier(c3)
        return y_pred


