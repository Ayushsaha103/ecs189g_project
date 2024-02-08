import pickle
import math
import matplotlib.pyplot as plt
import random


class Data:
    def __init__(self, path):
        self.path = path
        self.X_train, self.X_test, self.y_train, self.y_test = self.extractInfo()
        self.train_length, self.test_length = len(self.X_train), len(self.X_test)

    def extractInfo(self):
        X_train, X_test, y_train, y_test = [], [], [], []

        with  open(self.path, 'rb') as f:
            data = pickle.load(f)
            f.close()
            for instance in data['train']:
                X_train.append(instance['image'])
                y_train.append(instance['label'])
            for instance in data['test']:
                X_test.append(instance['image'])
                y_test.append(instance['label'])
        return X_train, X_test, y_train, y_test

    def get_train_test(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def showRandomImages(self, num_of_images, train=True):
        selected_images = self.X_train if train else self.X_test
        selected_labels = self.y_train if train else self.y_test
        selected_length = self.train_length if train else self.test_length
        index_samples = random.sample(range(0, selected_length), num_of_images)
        images = [selected_images[index] for index in index_samples]
        labels = [selected_labels[index] for index in index_samples]


        max_cols = 5  # Maximum number of columns
        max_image_size = 4  # Maximum size of each image in inches
        max_figsize = (max_image_size * max_cols, max_image_size * math.ceil(num_of_images / max_cols))

        num_cols = min(num_of_images, max_cols)
        num_rows = (num_of_images + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=max_figsize)
        axes = axes.flatten()


        for i, (label, image_data) in enumerate(zip(labels, images)):
            ax = axes[i]
            ax.imshow(image_data)
            ax.axis('off')
            ax.set_title(f'[{i}] Label: {label}')

        for i in range(num_of_images, num_rows * num_cols):
            axes[i].axis('off')

        plt.suptitle(f"{num_of_images} samples from {'training' if train else 'testing'} data")
        plt.tight_layout()
        plt.show()


