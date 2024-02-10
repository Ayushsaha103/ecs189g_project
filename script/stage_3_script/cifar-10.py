from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.stage_3_code.Method_CNN import Method_CNN
from code.stage_3_code.Result_Saver import Result_Saver
from code.stage_3_code.Setting_KFold_CV import Setting_KFold_CV
from code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
import numpy as np
import torch
from torch import nn
import itertools
import os

#################################################################################################################

# ---- parameter section -------------------------------
np.random.seed(2)
torch.manual_seed(2)
# ------------------------------------------------------


# ----model configs---

configurations = {
    'lr': [1e-3],
    'batch_size': [64],
    'loss_function': [nn.CrossEntropyLoss],
    'optimizer': [
        torch.optim.Adam
    ],
    "hidden_units": [10],
}

# params
input_shape = 3
output_shape = 10
output_layer_input_channels = 40

#


keys, values = zip(*configurations.items())
config_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

for config in config_permutations:

    print('Config: ', config)

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('cifar10_data', '')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    data_obj.dataset_source_file_name = 'CIFAR'

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = os.path.join('../../result/stage_3_result/cifar-10-model')

    if not os.path.exists(result_obj.result_destination_folder_path):
        os.makedirs(result_obj.result_destination_folder_path)

    result_obj.result_destination_file_name = 'MLP_prediction_result'

    method_obj = Method_CNN('multi-layer perceptron', '', result_obj.result_destination_folder_path, input_shape,
                            config['hidden_units'], output_shape, config['lr'], config['batch_size'],
                            config['loss_function'], config['optimizer'], max_epoch=50, output_layer_input_channels=output_layer_input_channels)

    setting_obj = Setting_KFold_CV('k fold cross validation', '')
    # setting_obj = Setting_Tra
    # in_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Metrics('metrics', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('CNN Accuracy: {:.2f}% +/- {:.2f}%.'.format(mean_score * 100, std_score * 100))
    print('************ Finish ************')
    # ------------------------------------------------------
