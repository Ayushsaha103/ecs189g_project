

from code.stage_2_code.Dataset_Loader import Dataset_Loader
from code.stage_2_code.Method_MLP import Method_MLP
from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_2_code.Setting_KFold_CV import Setting_KFold_CV
from code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
#
#
# loader = Dataset_Loader()
# loader.dataset_source_folder_path = "../../code/stage_2_code/stage_2_data/"
# loader.dataset_source_file_name = "train.csv"
# data = loader.load()
#
# loader2 = Dataset_Loader()
# loader2.dataset_source_folder_path = "../../code/stage_2_code/stage_2_data/"
# loader2.dataset_source_file_name = "test.csv"
# data2 = loader2.load()
#
# data = {"train": {"X": data["X"], "Y": data["y"]},
#         "test": {"X": data2["X"], "Y": data2["y"]}}
#
# method_obj = Method_MLP('multi-layer perceptron', '')
# method_obj.data = data
# method_obj.run()




#################################################################################################################
#

# ---- Multi-Layer Perceptron script ----
# ---- parameter section -------------------------------
np.random.seed(2)
torch.manual_seed(2)
# ------------------------------------------------------

# ---- objection initialization setction ---------------
data_obj = Dataset_Loader('train', '')
data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
data_obj.dataset_source_file_name = 'train.csv'

method_obj = Method_MLP('multi-layer perceptron', '')

result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
result_obj.result_destination_file_name = 'prediction_result'

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
print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
print('************ Finish ************')
# ------------------------------------------------------


