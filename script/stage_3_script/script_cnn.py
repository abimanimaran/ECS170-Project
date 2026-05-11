from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN import Method_CNN
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# =========================
# Reproducibility
# =========================
np.random.seed(2)
torch.manual_seed(2)

print("******** START ********")

# =========================
# Choose dataset
# =========================
dataset_name = 'ORL'   # MNIST / CIFAR / ORL

# Single dataset loader instance
data_obj = Dataset_Loader(dataset_name, '')
data_obj.dataset_source_folder_path = (
    '/Users/awsomehome/Desktop/ECS170_Spring_2026_Source_Code_Template/data/stage_3_data/'
)
data_obj.dataset_source_file_name = dataset_name

# Load data once
loaded_data = data_obj.load()
X_train, y_train = loaded_data['X_train'], loaded_data['y_train']
X_test, y_test = loaded_data['X_test'], loaded_data['y_test']



# =========================
# Method (CNN)
# =========================
method_obj = Method_CNN('CNN', '')

# =========================
# Result saver
# =========================
result_obj = Result_Saver('', '')
result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_'
result_obj.result_destination_file_name = 'prediction_result'

# =========================
# Setting
# =========================
setting_obj = Setting_Train_Test_Split('', '')
evaluate_obj = Evaluate_Accuracy('', '')

# Use the same dataset object
setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
setting_obj.print_setup_summary()

# =========================
# Run pipeline
# Pass data directly to method.run to avoid re-loading
metrics = method_obj.run(X_train, y_train, X_test, y_test)

# Save results
result_obj.save()

# =========================
# Compute additional metrics
# =========================
true_y = metrics['true_y']
pred_y = metrics['pred_y']

accuracy = accuracy_score(true_y, pred_y)
f1 = f1_score(true_y, pred_y, average='macro')
precision = precision_score(true_y, pred_y, average='macro')
recall = recall_score(true_y, pred_y, average='macro')

print("******** RESULT ********")
print(metrics)
print("\nCNN Test Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("******** FINISH ********")