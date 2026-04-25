'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
import torch
from torch import nn
import numpy as np

# ---- Method_MLP.py ----

class Method_MLP(method, nn.Module):
    data = None
    max_epoch = 70
    learning_rate = 1e-3

    def __init__(self, mName, mDescription):
        self.mName = mName
        self.mDescription = mDescription

        self.loss_history = []
        self.acc_history = []

        # Initialize nn.Module
        nn.Module.__init__(self)

        # Larger and deeper network
        self.fc_layer_1 = nn.Linear(784, 512)  # increased size
        self.activation_func_1 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(512, 256)
        self.activation_func_2 = nn.ReLU()
        self.fc_layer_3 = nn.Linear(256, 128)
        self.activation_func_3 = nn.ReLU()
        self.fc_layer_4 = nn.Linear(128, 10)

    def forward(self, x):
        h1 = self.activation_func_1(self.fc_layer_1(x))
        h2 = self.activation_func_2(self.fc_layer_2(h1))
        h3 = self.activation_func_3(self.fc_layer_3(h2))
        y_pred = self.fc_layer_4(h3)  # raw logits
        return y_pred

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()

        X_tensor = torch.FloatTensor(np.array(X))
        y_tensor = torch.LongTensor(np.array(y))

        evaluator = Evaluate_Metrics('train_eval', '')
        for epoch in range(self.max_epoch):
            optimizer.zero_grad()
            y_pred = self.forward(X_tensor)
            loss = loss_function(y_pred, y_tensor)
            loss.backward()
            optimizer.step()

            self.loss_history.append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == y_tensor).float().mean().item()
            self.acc_history.append(accuracy)

            if epoch % 5 == 0:
                evaluator.data = {
                    'true_y': y_tensor,
                    'pred_y': y_pred.argmax(dim=1)
                }
                metrics = evaluator.evaluate()
                print(f"Epoch {epoch}: Loss={loss.item():.4f}, "
                      f"Acc={metrics['accuracy']:.4f}, "
                      f"F1={metrics['f1_weighted']:.4f}, "
                      f"Precision={metrics['precision_weighted']:.4f}, "
                      f"Recall={metrics['recall_weighted']:.4f}")

    def test(self, X):


        X_tensor = torch.FloatTensor(np.array(X))
        y_pred = self.forward(X_tensor)

        return y_pred.argmax(dim=1)

    def run(self):
        print('-- Training MLP --')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('-- Testing MLP --')
        pred_y = self.test(self.data['test']['X'])

        evaluator = Evaluate_Metrics('test_eval', '')
        evaluator.data = {
            'true_y': torch.LongTensor(np.array(self.data['test']['y'])),
            'pred_y': pred_y
        }
        metrics = evaluator.evaluate()
        return metrics, pred_y
