import numpy as np

from code.base_class.dataset import dataset
import pickle
import os


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None, dSourceFolderPath=None, dSourceFileName=None,
                 shift_label_from_1_index_to_0_index=False, add_3rd_dimension=False):
        super().__init__(dName, dDescription)
        self.dataset_source_folder_path = dSourceFolderPath
        self.dataset_source_file_name = dSourceFileName
        self.add_3rd_dimension = add_3rd_dimension
        self.shift_label_from_1_index_to_0_index = shift_label_from_1_index_to_0_index
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
            }
        }

    def load(self):
        with open(os.path.join(self.dataset_source_folder_path, self.dataset_source_file_name), 'rb') as f:
            self.init_data()
            filedata = pickle.load(f)
            f.close()
            for key in self.data.keys():
                for instance in filedata[key]:
                    # pytorch is channel, height, width
                    image = np.transpose(instance['image'])
                    if self.add_3rd_dimension:
                        image = np.expand_dims(image, axis=0)
                    self.data[key]['X'].append(image)

                    self.data[key]['y'].append(
                        instance['label'] - (1 if self.shift_label_from_1_index_to_0_index else 0))

            return self.data
