'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from sklearn.model_selection import KFold
import numpy as np

class Setting_KFold_CV(setting):
    fold = 2
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()
        # train test here are from the seperate files

        train_file = loaded_data['train']
        test_file = loaded_data['test']
        
        kf = KFold(n_splits=self.fold, shuffle=True)
        
        fold_count = 0
        score_list = []
        for train_index, test_index in kf.split(train_file['X']):
            fold_count += 1
            print('************ Fold:', fold_count, '************')

            # train test here are both from train
            X_train, X_test = np.array(train_file['X'])[train_index], np.array(train_file['X'])[test_index]
            y_train, y_test = np.array(train_file['y'])[train_index], np.array(train_file['y'])[test_index]

            # test_file is from test file
            X_test_file = np.array(test_file['X'])
            Y_test_file = np.array(train_file['y'])

            # run MethodModule
            self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}, 'test_file': {'X': X_test_file, 'y': Y_test_file}}
            learned_result = self.method.run(fold_count)
            
            # save raw ResultModule
            self.result.data = learned_result
            self.result.fold_count = fold_count
            self.result.save()
            
            self.evaluate.data = learned_result
            score_list.append(self.evaluate.evaluate())
        
        return np.mean(score_list), np.std(score_list)

        