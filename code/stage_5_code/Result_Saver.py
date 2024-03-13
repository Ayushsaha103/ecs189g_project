import pickle
import os

class Result_Saver:
    def __init__(self, folder_path, file_name):
        self.folder_path = folder_path
        self.file_name = file_name

    def save(self, data):
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        with open(os.path.join(self.folder_path, self.file_name), 'wb') as file:
            pickle.dump(data, file)
