'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np

# ---- Setting_Train_Test_Split.py ----

class Setting_Train_Test_Split(setting):
    def load_run_save_evaluate(self):
        loaded_data = self.dataset.load()

        X_train = loaded_data['X_train']
        y_train = loaded_data['y_train']
        X_test = loaded_data['X_test']
        y_test = loaded_data['y_test']

        self.method.data = {'train': {'X': X_train, 'y': y_train},
                            'test': {'X': X_test, 'y': y_test}}

        # Run method (MLP)
        metrics = self.method.run()  # run() should return the dict directly

        # Save results
        self.result.data = metrics
        self.result.save()

        # No need to evaluate again
        return metrics