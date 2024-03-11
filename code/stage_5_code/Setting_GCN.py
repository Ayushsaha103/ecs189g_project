import torch

from code.base_class.setting import setting
import numpy as np


class Setting_GCN(setting):

    def load_run_save_evaluate(self):

        device = None

        # if torch.backends.mps.is_available():
        #     device = torch.device("mps")
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        print("Using device: ", device)

        self.method.to(device)

        # load dataset
        loaded_data = self.dataset.load()

        adj = loaded_data['graph']['utility']['A']
        features = loaded_data['graph']['X']
        labels = loaded_data['graph']['y']
        idx_train = loaded_data['train_test_val']['idx_train']
        # idx_val = loaded_data['train_test_val']['idx_val']
        idx_test = loaded_data['train_test_val']['idx_test']
        # train test here are from the seperate datas

        self.method.data = {'train': {'feature': features, 'adj': adj, 'label': labels, 'indx':idx_train},
                            'test': {'feature': features, 'adj': adj, 'label': labels, 'indx':idx_test}}

        learned_result = self.method.run()

        return learned_result