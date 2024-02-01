

from code.stage_2_code.Dataset_Loader import Dataset_Loader
from code.stage_2_code.Method_MLP import Method_MLP
from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_2_code.Setting_KFold_CV import Setting_KFold_CV
from code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
from torch import nn
import itertools
import os


#################################################################################################################
#

# ---- Multi-Layer Perceptron script ----
# ---- parameter section -------------------------------
np.random.seed(2)
torch.manual_seed(2)
# ------------------------------------------------------


#----model configs---

configurations = {
    'lr': [3e-4, 7e-4, 1e-4],
    'batch_size': [512, 1024, 2048],
    'loss_function': [nn.MSELoss, nn.L1Loss, nn.CrossEntropyLoss],
    'optimizer': [
        torch.optim.Adam,
        torch.optim.Adagrad,
        torch.optim.Adamax,
    ]
}

keys, values = zip(*configurations.items())
config_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

for config in config_permutations:

    print('Config: ', config)

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('train', '')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    data_obj.dataset_source_file_name = 'train.csv'

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = os.path.join('../../result/stage_2_result/', f'lr-{config["lr"]}/', f'batch-{config["batch_size"]}', f'loss_function-{config["loss_function"]}', f'optimizer-{config["optimizer"]}/')

    if not os.path.exists(result_obj.result_destination_folder_path):
        os.makedirs(result_obj.result_destination_folder_path)

    result_obj.result_destination_file_name = 'MLP_prediction_result'

    method_obj = Method_MLP('multi-layer perceptron', '', result_obj.result_destination_folder_path, config['lr'], config['batch_size'], config['loss_function'], config['optimizer'])

    setting_obj = Setting_KFold_CV('k fold cross validation', '')
    # setting_obj = Setting_Tra
    # in_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
#   print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('MLP Accuracy: {:.2f}% +/- {:.2f}%.'.format(mean_score * 100, std_score * 100))
    print('************ Finish ************')
    # ------------------------------------------------------




