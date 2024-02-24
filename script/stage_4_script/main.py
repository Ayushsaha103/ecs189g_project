# readme:
# this project does text classification. it iteratively stores segments of the entire data into a local dataframe 'df', then saves 'df' to a local file
# It trains by loading df back again into a torch tabular dataset, and training on what was loaded.

# current accuracy is ~53%





import torch
import torch.nn.functional as F
import torchtext
import time
import random
import pandas as pd
import os
torch.backends.cudnn.deterministic = True


### General Settings
RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)

VOCABULARY_SIZE = 10000
LEARNING_RATE = 0.005
BATCH_SIZE = 100
NUM_EPOCHS = 15
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

EMBEDDING_DIM = 300
HIDDEN_DIM = 340
NUM_CLASSES = 2

### RNN CLASS
class RNN(torch.nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        # self.rnn = torch.nn.RNN(embedding_dim,
        #                        hidden_dim,
        #                        nonlinearity='relu')
        self.rnn = torch.nn.LSTM(embedding_dim,
                                 hidden_dim)

        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text dim: [sentence length, batch size]

        embedded = self.embedding(text)
        # embedded dim: [sentence length, batch size, embedding dim]

        output, (hidden, cell) = self.rnn(embedded)
        # output dim: [sentence length, batch size, hidden dim]
        # hidden dim: [1, batch size, hidden dim]

        hidden.squeeze_(0)
        # hidden dim: [batch size, hidden dim]

        output = self.fc(hidden)
        return output



### Get Data
max_df_size = 5000
posd = '../../data/stage_4_data/text_classification/train/pos/'     # positive files directory
negd = '../../data/stage_4_data/text_classification/train/neg/'     # negative files directory
posfs = os.listdir(posd)        # list of pos filenames
negfs = os.listdir(negd)        # list of neg filenames

# func. that creates folder for the modified dataset segments to be stored
def create_modified_dataset_format_folder():
    if not os.path.exists('../../data/stage_4_data/text_classification/modified_format/'):
        directory = "modified_format"
        parent_dir = "../../data/stage_4_data/text_classification/"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)

# func. to read some positive files into file_contents
def read_pos_files(pos_cnter, file_contents, file_labels):
    for i in range(max_df_size // 2):
        file = open(posd + posfs[pos_cnter % len(posfs)], "r",  encoding="utf8")
        # print(file.read())
        file_contents.append(file.read())
        file_labels.append(1)
        pos_cnter += 1
    if pos_cnter > 999999: pos_cnter = 0        # reset pos_cnter
    return pos_cnter

# func. to read some negative files into file_contents
def read_neg_files(neg_cnter, file_contents, file_labels):
    for i in range(max_df_size // 2):
        file = open(negd + negfs[neg_cnter % len(negfs)], "r", encoding="utf8")
        # print(file.read())
        file_contents.append(file.read())
        file_labels.append(0)
        neg_cnter += 1
    if neg_cnter > 999999: neg_cnter = 0        # reset neg_cnter
    return neg_cnter

# driver func. to save reformatted dataset (.csv)
def reformat_and_save_dataset(updated_datasegfile, pos_cnter, neg_cnter):
    ### reformat and save dataset
    file_contents = []  # list of all strings in currently loaded dataset subset
    file_labels = []  # labels of all strings in currently loaded dataset subset

    pos_cnter = read_pos_files(pos_cnter, file_contents, file_labels)
    neg_cnter = read_neg_files(neg_cnter, file_contents, file_labels)

    df = pd.DataFrame({'Content': file_contents, 'Labels': file_labels})
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle dataframe

    print('read dataset segment:' + updated_datasegfile[updated_datasegfile.rfind('/'):])
    df.to_csv(updated_datasegfile, index=False)
    return pos_cnter, neg_cnter



pos_cnter = 0; neg_cnter = 0
create_modified_dataset_format_folder()
for i in range(5):

    ### reformat and save dataset
    updated_datasegfile = '../../data/stage_4_data/text_classification/modified_format/train_dataset_segment_' + str(i) + '.csv'
    pos_cnter, neg_cnter = reformat_and_save_dataset(updated_datasegfile, pos_cnter, neg_cnter)

    ### Defining the feature processing
    TEXT = torchtext.legacy.data.Field(
        tokenize='spacy',  # default splits on whitespace
        tokenizer_language='en_core_web_sm'
    )

    ### Defining the label processing
    LABEL = torchtext.legacy.data.LabelField(dtype=torch.long)
    # process the dataset
    fields = [('TEXT_COLUMN_NAME', TEXT), ('LABEL_COLUMN_NAME', LABEL)]
    dataset = torchtext.legacy.data.TabularDataset(
        path=updated_datasegfile, format='csv',
        skip_header=True, fields=fields)

    ### Split Dataset into Train/Validation/Test
    train_data, test_data = dataset.split(
        split_ratio=[0.8, 0.2],
        random_state=random.seed(RANDOM_SEED))
    print(f'Num Train: {len(train_data)}')
    print(f'Num Test: {len(test_data)}')

    train_data, valid_data = train_data.split(
        split_ratio=[0.85, 0.15],
        random_state=random.seed(RANDOM_SEED))
    print(f'Num Train: {len(train_data)}')
    print(f'Num Validation: {len(valid_data)}')
    # print(vars(train_data.examples[0]))



    ### Build Vocabulary
    # Build the vocabulary based on the top "VOCABULARY_SIZE" words:

    TEXT.build_vocab(train_data, max_size=VOCABULARY_SIZE)
    LABEL.build_vocab(train_data)

    # print(f'Vocabulary size: {len(TEXT.vocab)}')
    # print(f'Number of classes: {len(LABEL.vocab)}')

    # # print most common words
    # print(TEXT.vocab.freqs.most_common(20))

    # Define Data Loaders
    train_loader, valid_loader, test_loader = \
        torchtext.legacy.data.BucketIterator.splits(
            (train_data, valid_data, test_data),
             batch_size=BATCH_SIZE,
             sort_within_batch=False,
             sort_key=lambda x: len(x.TEXT_COLUMN_NAME),
             device=DEVICE
        )
    #
    print('Train')
    # for batch in train_loader:
    #     print(f'Text matrix size: {batch.TEXT_COLUMN_NAME.size()}')
    #     print(f'Target vector size: {batch.LABEL_COLUMN_NAME.size()}')
    #     break

    # print('\nValid:')
    # for batch in valid_loader:
    #     print(f'Text matrix size: {batch.TEXT_COLUMN_NAME.size()}')
    #     print(f'Target vector size: {batch.LABEL_COLUMN_NAME.size()}')
    #     break

    # print('\nTest:')
    # for batch in test_loader:
    #     print(f'Text matrix size: {batch.TEXT_COLUMN_NAME.size()}')
    #     print(f'Target vector size: {batch.LABEL_COLUMN_NAME.size()}')
    #     break



    torch.manual_seed(RANDOM_SEED)
    model = RNN(input_dim=len(TEXT.vocab),
                embedding_dim=EMBEDDING_DIM,
                hidden_dim=HIDDEN_DIM,
                output_dim=NUM_CLASSES  # could use 1 for binary classification
                )

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    ### Training


    def compute_accuracy(model, data_loader, device):

        with torch.no_grad():
            correct_pred, num_examples = 0, 0

            for i, (features, targets) in enumerate(data_loader):
                features = features.to(device)
                targets = targets.float().to(device)

                logits = model(features)
                _, predicted_labels = torch.max(logits, 1)

                num_examples += targets.size(0)
                correct_pred += (predicted_labels == targets).sum()
        return correct_pred.float() / num_examples * 100


    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_idx, batch_data in enumerate(train_loader):

            text = batch_data.TEXT_COLUMN_NAME.to(DEVICE)
            labels = batch_data.LABEL_COLUMN_NAME.to(DEVICE)

            ### FORWARD AND BACK PROP
            logits = model(text)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()

            loss.backward()

            ### UPDATE MODEL PARAMETERS
            optimizer.step()

            ### LOGGING
            if 1 == 1:      # not batch_idx % 50:
                print(f'Epoch: {epoch + 1:03d}/{NUM_EPOCHS:03d} | '
                      f'Batch {batch_idx:03d}/{len(train_loader):03d} | '
                      f'{compute_accuracy(model, train_loader, DEVICE):.2f}% | '
                      f'Loss: {loss:.4f}')

        with torch.set_grad_enabled(False):
            print(f'training accuracy: '
                  f'{compute_accuracy(model, train_loader, DEVICE):.2f}%'
                  f'\nvalid accuracy: '
                  f'{compute_accuracy(model, valid_loader, DEVICE):.2f}%')

        print(f'Time elapsed: {(time.time() - start_time) / 60:.2f} min')

    print(f'Total Training Time: {(time.time() - start_time) / 60:.2f} min')
    print(f'Test accuracy: {compute_accuracy(model, test_loader, DEVICE):.2f}%')






# # load model from saved checkpoint file
# def load_ckp(checkpoint_fpath, model, optimizer):
#     # reload the checkpoint, model, and optimizer
#     checkpoint = torch.load(checkpoint_fpath)
#     model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#
#     # reload the weights and biases (i think this works, hopefully)
#     state_dict = model.state_dict()
#     avoid = ['fc.weight', 'fc.bias']
#     for key in checkpoint.keys():
#         if key in avoid or key not in state_dict.keys():
#             continue
#         if checkpoint[key].size() != state_dict[key].size():
#             continue
#         state_dict[key] = checkpoint[key]
#     model.load_state_dict(state_dict)
#
#     return model, optimizer, checkpoint['epoch']
#
# checkpoint = {
#     'epoch': epoch + 1,
#     'state_dict': model.state_dict(),
#     'optimizer': optimizer.state_dict()
# }
# checkpoint_fpath = '../../result/stage_4_result/' + 'classification_checkpoint.pt'
# torch.save(checkpoint, checkpoint_fpath)
# model, optimizer, start_epoch = load_ckp(checkpoint_fpath, model, optimizer)
#
