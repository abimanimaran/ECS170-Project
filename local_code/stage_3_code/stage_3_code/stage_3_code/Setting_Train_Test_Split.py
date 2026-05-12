'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np

class Setting_Train_Test_Split(setting):
    fold = 3

    def load_run_save_evaluate(self):
        loaded_data = self.dataset.load()

        self.method.data = {
            'train': {
                'X': loaded_data['X_train'],
                'y': loaded_data['y_train']
            },
            'test': {
                'X': loaded_data['X_test'],
                'y': loaded_data['y_test']
            }
        }

        data = self.dataset.load()

        learned_result = self.method.run(
            data['X_train'],
            data['y_train'],
            data['X_test'],
            data['y_test']
        )
        # save results
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = {
            'true_y': learned_result['true_y'],
            'pred_y': learned_result['pred_y']
        }

        score = self.evaluate.evaluate()

        return score