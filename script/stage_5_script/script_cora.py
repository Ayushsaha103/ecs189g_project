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
from code.stage_5_code.pygcn.models import GCN
from code.stage_5_code.GCN_cora import GCN_cora
from code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader

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

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Get data path from command line arguments
data_path = "../../data/stage_5_data/cora"

# Load data using load_data method
# adj, features, labels, idx_train, idx_test = load_data(path=data_path, dataset=args.dataset)
data_obj = Dataset_Loader(args.dataset, '')
data_obj.dataset_source_folder_path = os.path.join(args.data_folder, args.dataset)
data_obj.dataset_name = args.dataset
D = data_obj.load()

adj = D['graph']['utility']['A']
features = D['graph']['X']
labels = D['graph']['y']
idx_train = D['train_test_val']['idx_train']
idx_test = D['train_test_val']['idx_test']

# Model and optimizer
model = GCN_cora(nfeat=features.shape[1],
                 nhid=args.hidden,
                 nclass=labels.max().item() + 1,
                 dropout_rate=args.dropout)

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


# Train model
def train_model():
    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch)
        loss_test, acc_test, precision, recall, f1 = test()  # Call test here
        test_losses.append(loss_test)                        # Save test metrics after each epoch
        test_accuracies.append(acc_test)
        test_precisions.append(precision)
        test_recalls.append(recall)
        test_f1_scores.append(f1)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


# Train model and test
t_total = time.time()
for epoch in range(args.epochs):
    train_loss, train_acc = train(epoch)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    test_loss, test_acc, test_precision, test_recall, test_f1 = test()
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    test_precisions.append(test_precision)
    test_recalls.append(test_recall)
    test_f1_scores.append(test_f1)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Plotting
# Plot training loss and accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(train_accuracies, label='Train Accuracy')
plt.title('Training Loss & Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Plot testing metrics
plt.subplot(1, 2, 2)
plt.plot(test_losses, label='Test Loss')
plt.plot(test_accuracies, label='Test Accuracy')
plt.plot(test_precisions, label='Test Precision')
plt.plot(test_recalls, label='Test Recall')
plt.plot(test_f1_scores, label='Test F1 Score')
plt.title('Testing Metrics')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()