import torch

from code.base_class.setting import setting
from sklearn.model_selection import KFold
import numpy as np


class SettingJokeGeneration(setting):

    def load_train(self):

        device = None

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        print("Using device: ", device)

        self.method.to(device)

        # load dataset
        loaded_data = self.dataset.load()

        X = loaded_data['X']
        y = loaded_data['y']

        print(12)
        # run MethodModule
        self.method.data = {'train': {'X': X, 'y': y}}

        learned_result = self.method.run(0)

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        return learned_result
