import pickle
import random

import numpy as np

from code.base_class.dataset import dataset
import re
import os

import nltk
from nltk.corpus import stopwords




class Generation_Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None, dSourceFilePath=None, total_files_amount=1622,
                 cache_file_name=None):
        super().__init__(dName, dDescription)
        self.dataset_source_file_path = dSourceFilePath
        # word_dic is map between word and index
        self.total_files_amount = total_files_amount
        self.cache_file_name = cache_file_name
        self.already_loaded = False
        self.init_data()

    def init_data(self):
        self.data = {
            'sentences': [],
            'X': [],
            'y': [],
            'word_dic': {'<padding>': 0, "<end>": 1},
            'word_dic_rev': {}
        }

    def preprocess_text(self, text):
        lower_text = text.lower()
        punctuation = r'!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'
        allowed_punctuation = r".?"
        punctuation_enhanced_text = ''
        for c in lower_text:
            if c not in punctuation:
                punctuation_enhanced_text += c
            elif c in allowed_punctuation:
                punctuation_enhanced_text += f" {c} "
        sentence_words = punctuation_enhanced_text.split(' ')
        sentence_words = [ *sentence_words, '<end>']
        return sentence_words

    def sentence_to_word_dic_encoded(self, sentence):
        result_sentence = []
        for word in sentence:
            if word not in self.data['word_dic']:
                self.data['word_dic'][word] = len(self.data['word_dic'])
            result_sentence.append(self.data['word_dic'][word])
        return result_sentence


    def build_dataset_from_files(self):
        i = 0
        longest_sentence_length = 0
        with open(self.dataset_source_file_path, 'r') as f:
            lines = f.read().splitlines()

        for line in lines[1:]:
            sentence = self.preprocess_text(line.split(',')[1].strip())
            word_dic_encoded = self.sentence_to_word_dic_encoded(sentence)
            longest_sentence_length = max(longest_sentence_length, len(word_dic_encoded))
            self.data['sentences'].append(word_dic_encoded)
            i += 1
            if i % 100 == 0 or i == self.total_files_amount:
                print("Loading Data: {}/{}".format(i, self.total_files_amount))

        for sentence in self.data['sentences']:

            # split into 3 word groups
            for j in range(0, len(sentence)-3):
                x = [sentence[j], sentence[j+1], sentence[j+2]]
                y = sentence[j+3]
                # y = [sentence[j+1], sentence[j+2], sentence[j+3]]

                encoded_y = np.zeros(len(self.data['word_dic']))
                encoded_y[y] = 1

                self.data['X'].append(x)
                self.data['y'].append(encoded_y)

        self.data['X'] = np.array(self.data['X'])
        self.data['y'] = np.array(self.data['y'])



    def create_rev_word_dic(self):
        self.data['word_dic_rev'] = dict(map(reversed, self.data['word_dic'].items()))

    def load(self):
        if not self.already_loaded:
            if os.path.exists(self.cache_file_name):
                print("Loading from cache")
                self.data = pickle.load(open(self.cache_file_name, 'rb'))
            else:
                print("Building dataset from files")
                self.build_dataset_from_files()
                self.create_rev_word_dic()
                pickle.dump(self.data, open(self.cache_file_name, 'wb+'))

        self.already_loaded = True
        return self.data
