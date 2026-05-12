'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score

class Evaluate_Accuracy:
    def __init__(self, dummy1, dummy2):
        self.data = {}
        self.evaluate_name = 'Accuracy'

    def evaluate(self):
        return accuracy_score(self.data['true_y'], self.data['pred_y'])