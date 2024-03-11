

import sys
sys.path.append("../../code/stage_5_code/")


dataset_name = 'citeseer'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader

data_obj = Dataset_Loader(dataset_name, '')
data_obj.dataset_source_folder_path = '../../data/stage_5_data/' + dataset_name
data_obj.dataset_name = dataset_name

D = data_obj.load()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from code.stage_5_code.myGCN_driver import GCN_driver

g = GCN_driver(epochs_=250, lr_=0.003, wt_decay=0.003, nhidden_=12, dropout_=0.5)
g.load_data(D, dataset_name)
g.train()
g.test()
