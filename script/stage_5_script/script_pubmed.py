import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from code.stage_5_code.pygcn.utils import accuracy
from code.stage_5_code.GCN_citeseer import GCN_citeseer
from code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
import random

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, default='../../data/stage_5_data/',
                    help='Path to the data folder')
parser.add_argument('--dataset', type=str, default='cora',
                    help='Name of the dataset')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')


# Set a fixed random seed for reproducibility
def set_seed(seed_value):
    random.seed(seed_value)  # Python's built-in random module
    np.random.seed(seed_value)  # Numpy library
    torch.manual_seed(seed_value)  # PyTorch library
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # Environment variable

    # If using a CUDA-enabled GPU, the following two lines are needed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Set the seed value
set_seed(42)

# Set hyperparameters
parser.set_defaults(lr=0.01)
parser.set_defaults(weight_decay=5e-4)
parser.set_defaults(hidden=16)
parser.set_defaults(dropout=0.5)
parser.set_defaults(layers=2)
parser.set_defaults(patience=15)
parser.set_defaults(batch_size=32)
parser.set_defaults(dataset='pubmed')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Get data path from command line arguments
data_path = "../../data/stage_5_data/pubmed"  # Adjust path for PubMed

# Load data using load_data method
data_obj = Dataset_Loader('pubmed', '')  # Adjust for PubMed
data_obj.dataset_source_folder_path = os.path.join(args.data_folder, 'pubmed')
data_obj.dataset_name = 'pubmed'
D = data_obj.load()

# Adjust the feature and class count for PubMed
adj = D['graph']['utility']['A']
features = D['graph']['X']
labels = D['graph']['y']
idx_train = D['train_test_val']['idx_train']
idx_test = D['train_test_val']['idx_test']

# Model and optimizer
model = GCN_citeseer(nfeat=500,  # PubMed has 500 features per node
                     nhid=args.hidden,
                     nclass=3,  # PubMed has 3 classes
                     dropout_rate=args.dropout,
                     layers=args.layers)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_test = idx_test.cuda()

# Initialize lists to store loss, accuracy for plotting
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
test_precisions = []
test_recalls = []
test_f1_scores = []

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(features, adj)

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_train.item(), acc_train.item()


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    preds = output[idx_test].max(1)[1].type_as(labels[idx_test])
    precision = precision_score(labels[idx_test].cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)
    recall = recall_score(labels[idx_test].cpu().numpy(), preds.cpu().numpy(), average='macro')
    f1 = f1_score(labels[idx_test].cpu().numpy(), preds.cpu().numpy(), average='macro')
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()),
          "precision= {:.4f}".format(precision),
          "recall= {:.4f}".format(recall),
          "F1 score= {:.4f}".format(f1))
    return loss_test.item(), acc_test.item(), precision, recall, f1


# Early Stopping parameters
patience = args.patience  # How many epochs to wait after last time test loss improved.
patience_counter = 0  # Counter for Early Stopping

# Train model
def train_model():
    # Initialize the minimum test loss for early stopping
    test_loss_min = float('inf')

    t_total = time.time()
    for epoch in range(args.epochs):
        train_loss, train_acc = train(epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Get the test loss for early stopping
        test_loss, test_acc, test_precision, test_recall, test_f1 = test()
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        test_f1_scores.append(test_f1)

        # Check if test loss improved
        if test_loss < test_loss_min:
            test_loss_min = test_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Stopping early due to no improvement in test loss.")
            break

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


# Call train_model instead of the for loop
train_model()

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(20, 8))  # Keep the figsize to make plots larger
fig.suptitle(f'Dataset: {args.dataset} | LR: {args.lr} | Loss: NLLLoss | Optimizer: Adam', fontsize=16)

# Plot training loss and accuracy
axs[0].plot(train_losses, label='Train Loss')
axs[0].plot(train_accuracies, label='Train Accuracy')
axs[0].set_title('Training Loss & Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].legend()

# Plot testing metrics
axs[1].plot(test_losses, label='Test Loss')
axs[1].plot(test_accuracies, label='Test Accuracy')
axs[1].plot(test_precisions, label='Test Precision')
axs[1].plot(test_recalls, label='Test Recall')
axs[1].plot(test_f1_scores, label='Test F1 Score')
axs[1].set_title('Testing Metrics')
axs[1].set_xlabel('Epoch')
axs[1].legend()

# Adjust the layout to make room for the suptitle and prevent overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# Reduce horizontal space between the plots
plt.subplots_adjust(wspace=0.1)  # Reduced wspace value

plt.show()
