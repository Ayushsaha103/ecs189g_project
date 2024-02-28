
from code.stage_4_code.Classification_Dataset_Loader import Classification_Dataset_Loader
from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_4_code.Method_Classification_Rnn import Method_Classification_RNN
from code.stage_4_code.Method_Classification_LSTM import Method_Classification_LSTM
from code.stage_4_code.Setting_Movie_Classification import Setting_Movie_Classification
from code.stage_4_code.Evaulate_Classification_Metrics import Evaluate_Classification_Metrics
import os
import torch


### Configs

embedding_size = 7
hidden_size=64
output_size = 1
learning_rate = 0.001
batch_size = 4096

#num_rnn_layers = 7
num_lstm_layers = 2

#loss_function = torch.nn.BCELoss
loss_function = torch.nn.BCEWithLogitsLoss  # LSTM

optimizer = torch.optim.Adam
max_epochs = 100

###

data_obj = Classification_Dataset_Loader('classification_data', '')
data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_classification'
data_obj.cache_file_name="../../data/stage_4_data/text_classification/cache.pickle"

data_obj.load()

vocab_size = len(data_obj.data['word_dic'])

print("vocab length:", vocab_size)
sampleX = data_obj.data['train']['X'][20]
sampleY = data_obj.data['train']['y'][20]
sampleXInWords = [data_obj.data['word_dic_rev'][x] for x in sampleX]
print("sample X", sampleX, "sample Y", sampleY, 'sample X in words', sampleXInWords, sep='\n')

print(len(sampleX))

sampleX = data_obj.data['test']['X'][25]
sampleY = data_obj.data['test']['y'][25]

print(len(sampleX))

sampleXInWords = [data_obj.data['word_dic_rev'][x] for x in sampleX]
print("sample X", sampleX, "sample Y", sampleY, 'sample X in words', sampleXInWords, sep='\n')


text_length = len(sampleX)

#
result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = os.path.join('../../result/stage_4_result/classification')

if not os.path.exists(result_obj.result_destination_folder_path):
    os.makedirs(result_obj.result_destination_folder_path)

#result_obj.result_destination_file_name = 'RNN_classification_result'
result_obj.result_destination_file_name = 'LSTM_classification_result'

#method_obj = Method_Classification_RNN("Classification RNN", '', result_obj.result_destination_folder_path, vocab_size, embedding_size, hidden_size, num_rnn_layers, output_size, learning_rate, batch_size, loss_function, optimizer, max_epochs, text_length=text_length)
method_obj = Method_Classification_LSTM("Classification LSTM", '', result_obj.result_destination_folder_path, vocab_size, embedding_size, hidden_size, num_lstm_layers, output_size, learning_rate, batch_size, loss_function, optimizer, max_epochs, text_length=text_length)


setting_obj = Setting_Movie_Classification('classification', '')

evaluate_obj = Evaluate_Classification_Metrics('metrics', '')


# # ---- running section ---------------------------------
print('************ Start ************')
setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
setting_obj.print_setup_summary()
result = setting_obj.load_run_save_evaluate()
print('************ Overall Performance ************')
print(result)
print('************ Finish ************')