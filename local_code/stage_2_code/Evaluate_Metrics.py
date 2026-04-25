from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class Evaluate_Metrics:
    def __init__(self, name, desc):
        self.name = name
        self.desc = desc
        self.data = None

    def evaluate(self, average='weighted', zero_division=0):
        import numpy as np

        y_true = self.data['true_y']
        y_pred = self.data['pred_y']

        # Convert tensors to numpy if needed
        if hasattr(y_true, 'detach'):
            y_true = y_true.detach().cpu().numpy()
        if hasattr(y_pred, 'detach'):
            y_pred = y_pred.detach().cpu().numpy()

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_weighted': f1_score(y_true, y_pred, average=average, zero_division=zero_division),
            'precision_weighted': precision_score(y_true, y_pred, average=average, zero_division=zero_division),
            'recall_weighted': recall_score(y_true, y_pred, average=average, zero_division=zero_division)
        }

        return metrics