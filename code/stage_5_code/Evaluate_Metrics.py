'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class Evaluate_Metrics(evaluate):
    data = None

    def evaluate(self, average='weighted'):
        true_labels = self.data['true_y']
        predicted_labels = self.data['pred_y']
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average=average, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, average=average, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, average=average, zero_division=0)
        return accuracy, precision, recall, f1

    def generate_confusion_matrix(self):
        true_labels = self.data['true_y']
        predicted_labels = self.data['pred_y']
        cm = confusion_matrix(true_labels, predicted_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(ax=ax)
        plt.title('Confusion Matrix')
        plt.show()
