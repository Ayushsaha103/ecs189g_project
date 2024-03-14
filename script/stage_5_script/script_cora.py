import os
import numpy as np
import torch
import torch.optim as optim
from code.stage_5_code.Method_GCN import Method_GCN
from code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from code.stage_5_code.Evaluate_Metrics import Evaluate_Metrics
import random

# ---- parameter section -------------------------------
np.random.seed(42)
torch.manual_seed(42)
# ------------------------------------------------------

# ---- model and training configs ---
configurations = {
    'lr': 0.01,
    'weight_decay': 5e-4,
    'hidden': 32,
    'dropout': 0.5,
    'layers': 2,
    'patience': 35,
    'epochs': 150
}

# ---- objection initialization section ---------------
dataset_name = 'cora'
data_obj = Dataset_Loader(dataset_name, '')
data_obj.dataset_source_folder_path = os.path.join('../../data/stage_5_data/', dataset_name)
data_obj.dataset_name = dataset_name
D = data_obj.load()

evaluate_obj = Evaluate_Metrics('evaluator', '')

# ---- running section ---------------------------------
print('************ Start ************')

adj = D['graph']['utility']['A']
features = D['graph']['X']
labels = D['graph']['y']
idx_train = D['train_test_val']['idx_train']
idx_test = D['train_test_val']['idx_test']

# Model and optimizer
model = Method_GCN(mName='GCN',
                   mDescription='Graph Convolutional Network',
                   nfeat=features.shape[1],
                   nhid=configurations['hidden'],
                   nclass=labels.max().item() + 1,
                   epochs=configurations['epochs'],
                   dropout_rate=configurations['dropout'],
                   layers=configurations['layers'],
                   learning_rate=configurations['lr'],
                   weight_decay=configurations['weight_decay'])

# Set model attributes for data
model.features = features
model.adj = adj
model.labels = labels
model.idx_train = idx_train
model.idx_test = idx_test

optimizer = optim.Adam(model.parameters(),
                       lr=configurations['lr'], weight_decay=configurations['weight_decay'])
model.optimizer = optimizer  # Set optimizer as an attribute of the model

# CUDA check and model.to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.features = model.features.to(device)
model.adj = model.adj.to(device)
model.labels = model.labels.to(device)
model.idx_train = model.idx_train.to(device)
model.idx_test = model.idx_test.to(device)

# Train model
model.train_model()

# Test model
test_loss, test_acc, precision, recall, f1 = model.test()

# Visualization
model.plot_metrics(dataset_name=dataset_name, learning_rate=model.args.lr)

print('************ Finish ************')