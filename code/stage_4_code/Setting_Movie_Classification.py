import torch

from code.base_class.setting import setting
from sklearn.model_selection import KFold
import numpy as np


class Setting_Movie_Classification(setting):

    def load_run_save_evaluate(self):

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

        train_data = loaded_data['train']
        test_data = loaded_data['test']

        X_train, X_test = np.array(train_data['X']), np.array(test_data['X'])
        y_train, y_test = np.array(train_data['y']), np.array(test_data['y'])

        print(y_train[0])

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train},
                            'test_data': {'X': X_test, 'y': y_test}}

        learned_result = self.method.run(0)

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        return learned_result
