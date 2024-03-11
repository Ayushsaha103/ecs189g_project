from code.stage_5_code.Dataset_Loader import Dataset_Loader
from code.stage_5_code.Method_GCN import GCN
from code.stage_5_code.Result_Saver import Result_Saver
from code.stage_5_code.Setting_GCN import Setting_GCN
from code.stage_5_code.Evaluate_Metrics import Evaluate_Metrics
import numpy as np
import torch
from torch import nn
import itertools
import os
from code.stage_5_code.pygcn.utils import load_data, accuracy
# from code.stage_5_code.pygcn.models import GCN
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import warnings

# Filter out the specific warning
warnings.filterwarnings("ignore", message="Converting sparse tensor to CSR format*")

#################################################################################################################

# ---- parameter section -------------------------------
np.random.seed(2)
torch.manual_seed(2)
# ------------------------------------------------------


# ----model configs---

configurations = {
    'lr': [1e-2],
    'loss_function': [nn.NLLLoss],
    'optimizer': [
        torch.optim.Adam
    ],
    "hidden_units": [64],
    "dropout": [0.6],
}

# params
max_epoch = 50
#


keys, values = zip(*configurations.items())
config_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

for config in config_permutations:

    print('Config: ', config)

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('citeseer', '')
    data_obj.dataset_source_folder_path = '../../data/stage_5_data/citeseer'
    data_obj.dataset_name = 'citeseer'
    D = data_obj.load()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # # Load data
    # adj, features, labels, idx_train, idx_val, idx_test = load_data()

    adj = D['graph']['utility']['A']
    features = D['graph']['X']
    labels = D['graph']['y']
    idx_train = D['train_test_val']['idx_train']
    idx_test = D['train_test_val']['idx_test']

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = os.path.join('../../result/stage_5_result/citeseer')

    if not os.path.exists(result_obj.result_destination_folder_path):
        os.makedirs(result_obj.result_destination_folder_path)

    result_obj.result_destination_file_name = 'GCN_prediction_result'


    method_obj = GCN('GCN', '',
                     result_obj.result_destination_folder_path,
                     nfeat=features.shape[1],
                     nhid=config['hidden_units'],
                     nclass=labels.max().item() + 1,
                     dropout=config['dropout'])

    setting_obj = Setting_GCN()

    evaluate_obj = Evaluate_Metrics('metrics', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    result = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print(f'Accuracy: {result["accuracy"]:.2f}')
    print('************ Finish ************')
    # ------------------------------------------------------
