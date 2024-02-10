from code.base_class.setting import setting
from sklearn.model_selection import KFold
import numpy as np


class Setting_KFold_CV(setting):
    fold = 2

    def load_run_save_evaluate(self):
        # load dataset
        loaded_data = self.dataset.load()
        # train test here are from the seperate datas
        train_data = loaded_data['train']
        test_data = loaded_data['test']

        kf = KFold(n_splits=self.fold, shuffle=True)

        fold_count = 0
        score_list = []
        for train_index, test_index in kf.split(train_data['X']):
            fold_count += 1
            print('************ Fold:', fold_count, '************')

            # train test here are both from train
            X_train, X_test = np.array(train_data['X'])[train_index], np.array(train_data['X'])[test_index]
            y_train, y_test = np.array(train_data['y'])[train_index], np.array(train_data['y'])[test_index]

            # test_data is from test data
            X_test_data = np.array(test_data['X'])
            Y_test_data = np.array(test_data['y'])

            # run MethodModule
            self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test},
                                'test_data': {'X': X_test_data, 'y': Y_test_data}}
            learned_result = self.method.run(fold_count)

            # save raw ResultModule
            self.result.data = learned_result
            self.result.fold_count = fold_count
            self.result.save()


            score_list.append(learned_result['accuracy'])

        return np.mean(score_list), np.std(score_list)
