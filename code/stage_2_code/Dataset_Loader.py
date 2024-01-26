'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    train_dataset_source_folder_path = None
    train_dataset_source_file_name = None

    test_dataset_source_folder_path = None
    test_dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        # cache
        if self.data is not None:
            return self.data

        print('loading data...')
        self.data = {}
        for data_type in ['train', 'test']:
            X = []
            y = []
            f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r')
            for line in f:
                line = line.strip('\n')
                elements = [int(i) for i in line.split(',')]
                X.append(elements[:-1])
                y.append(elements[-1])
            f.close()
            self.data[data_type] = {'X': X, 'y': y}

        return self.data
