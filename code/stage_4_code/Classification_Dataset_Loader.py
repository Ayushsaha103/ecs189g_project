import pickle
import random

import numpy as np

from code.base_class.dataset import dataset
import re
import os

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = stopwords.words('english')

from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()


class Classification_Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None, dSourceFolderPath=None, total_files_amount=50000,
                 cache_file_name=None):
        super().__init__(dName, dDescription)
        self.dataset_source_folder_path = dSourceFolderPath
        # word_dic is map between word and index
        self.total_files_amount = total_files_amount
        self.cache_file_name = cache_file_name
        self.already_loaded = False
        self.init_data()

    def init_data(self):
        self.data = {
            'train': {
                "X": [],
                "y": [],
            },
            'test': {
                'X': [],
                'y': []
            },

            'word_dic': {'<padding>': 0},
            'word_dic_rev': {}
        }

    def preprocess_text(self, text):
        lower_text = text.lower()
        punctuation = r'!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'
        punctuation_removed_text = ''
        for c in lower_text:
            if c not in punctuation:
                punctuation_removed_text += c
        sentence_words = punctuation_removed_text.split(' ')
        cleaned_words = [porter.stem(word) for word in sentence_words if word.isalpha() and (word not in stop_words)]
        trunced_sentence = cleaned_words[:55]
        return trunced_sentence

    def sentence_to_word_dic_encoded(self, sentence):
        result_sentence = []
        for word in sentence:
            if word not in self.data['word_dic']:
                self.data['word_dic'][word] = len(self.data['word_dic'])
            result_sentence.append(self.data['word_dic'][word])
        return result_sentence

    def pad_sentence(self, sentence, length):
        padded_sentence = sentence
        for i in range(0, length-len(sentence)):
            padded_sentence.append(self.data['word_dic']['<padding>'])
        return padded_sentence

    def build_dataset_from_files(self):
        file_categories = {
            "pos": 0,
            "neg": 1,
        }
        i = 0
        longest_sentence_length = 0
        groups = ('train', 'test')
        for group in groups:
            for dir, category_index in file_categories.items():
                file_dir = os.path.join(self.dataset_source_folder_path, group, dir)

                for filename in os.listdir(file_dir):
                    with open(os.path.join(file_dir, filename), 'r') as f:
                        sentence = self.preprocess_text(f.read())
                    word_dic_encoded = self.sentence_to_word_dic_encoded(sentence)
                    longest_sentence_length = max(longest_sentence_length, len(word_dic_encoded))
                    self.data[group]['X'].append(word_dic_encoded)
                    self.data[group]['y'].append(category_index)
                    i += 1
                    if i % 100 == 0:
                        print("Loading Data: {}/{}".format(i, self.total_files_amount))

        for group in groups:
            self.data[group]['X'] = [self.pad_sentence(x, longest_sentence_length) for x in self.data[group]['X']]
            combined = list(zip(self.data[group]['X'], self.data[group]['y']))
            random.shuffle(combined)
            self.data[group]['X'], self.data[group]['y'] = zip(*combined)



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
