from code.stage_4_code.Generation_Dataset_Loader import Generation_Dataset_Loader
import torch
import numpy as np
from numpy.random import choice

#######
# 3 words to start with, must be in the dictionary

start_words = ["what", "did", "the"]
max_length = 50

#######

save_path = "../../result/stage_4_result/generation/model.pt"

data_obj = Generation_Dataset_Loader('generation data', '')
data_obj.dataset_source_file_path = '../../data/stage_4_data/text_generation/data'
data_obj.cache_file_name="../../data/stage_4_data/text_generation/cache.pickle"
data_obj.load()
vocab_size = len(data_obj.data['word_dic'])


model = torch.load(save_path)
res = model
print(model)

encoded_start_words = [data_obj.data['word_dic'][x] for x in start_words]
print(start_words, encoded_start_words)

# print(data_obj.data['X'][:30])


input = encoded_start_words

result = start_words
hidden = None

for i in range(max_length):
    generation, _ = model.generate([input], hidden)
    out =  np.squeeze(generation.cpu().numpy())
    print(out)
    # next_in = choice(range(0, vocab_size), p=out/out.sum())
    next_in = np.argmax(out)
    next_in_in_words = data_obj.data['word_dic_rev'][next_in]



    result.append(next_in_in_words)

    input = [input[1], input[2], next_in]
    input_in_words = [data_obj.data['word_dic_rev'][x] for x in input]

    print(next_in, next_in_in_words, input, input_in_words)

    if next_in_in_words== '<end>':
        break


print(' '.join(result))