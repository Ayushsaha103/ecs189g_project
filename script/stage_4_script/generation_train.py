from code.stage_4_code.Generation_Dataset_Loader import Generation_Dataset_Loader
from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_4_code.Setting_Joke_Generation import SettingJokeGeneration
from code.stage_4_code.Method_Generation_Rnn import MethodGenerationRNN
from code.stage_4_code.Evaluate_Generation_Metrics import Evaluate_Generation_Metrics
import numpy as np
import os, torch

### Configs

embedding_size = 20
hidden_size=512
output_size = 1
learning_rate = 0.001
batch_size = 256
num_rnn_layers = 6
loss_function = torch.nn.CrossEntropyLoss
optimizer = torch.optim.Adam
max_epochs = 200
text_length=3
save_path = "../../result/stage_4_result/generation/model.pt"
###

data_obj = Generation_Dataset_Loader('generation data', '')
data_obj.dataset_source_file_path = '../../data/stage_4_data/text_generation/data'
data_obj.cache_file_name="../../data/stage_4_data/text_generation/cache.pickle"

data_obj.load()

vocab_size = len(data_obj.data['word_dic'])

print("vocab length:", vocab_size)
sampleSentence = data_obj.data['sentences'][0]
sampleSentenceInWords = [data_obj.data['word_dic_rev'][x] for x in sampleSentence]
sampleX = data_obj.data['X'][0]
sampleY = data_obj.data['y'][0]
print(np.argmax(sampleY))
sampleXInWords = [data_obj.data['word_dic_rev'][x] for x in sampleX]
print("sample sentence", sampleSentence, 'sample sentence in words', sampleSentenceInWords, sep='\n')
print("sample X", sampleX, "sample Y", sampleY, 'sample X in words', sampleXInWords, sep='\n')


result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = os.path.join('../../result/stage_4_result/generation')

if not os.path.exists(result_obj.result_destination_folder_path):
    os.makedirs(result_obj.result_destination_folder_path)

result_obj.result_destination_file_name = 'RNN_generation_result'

method_obj = MethodGenerationRNN("Generation RNN", '', result_obj.result_destination_folder_path, vocab_size, embedding_size, hidden_size, num_rnn_layers, output_size, learning_rate, batch_size, loss_function, optimizer, max_epochs, text_length=text_length, save_path=save_path)

# method_obj=torch.load(save_path)
# method_obj.max_epoch = max_epochs
#
metrics_obj = Evaluate_Generation_Metrics()

setting_obj = SettingJokeGeneration('generation rnn', '')


# # ---- running section ---------------------------------
print('************ Start ************')
setting_obj.prepare(data_obj, method_obj, result_obj, metrics_obj)
setting_obj.print_setup_summary()
result = setting_obj.load_train()
print('************ Overall Performance ************')
print(result)
print('************ Finish ************')