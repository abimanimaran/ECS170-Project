import matplotlib.pyplot as plt
from local_code.stage_3_code.Method_CNN import Method_CNN
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import numpy as np
import random

# =========================
# Set seeds for reproducibility
# =========================
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# =========================
# Dataset setup
# =========================
dataset_name = 'ORL'   # MNIST / CIFAR / ORL
data_obj = Dataset_Loader(dataset_name, '')
data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
data_obj.dataset_source_file_name = dataset_name

# =========================
# Load data only once
# =========================
print("loading data...")
loaded_data = data_obj.load()
print("data loaded successfully!")

X_train, y_train = loaded_data['X_train'], loaded_data['y_train']
X_test, y_test = loaded_data['X_test'], loaded_data['y_test']
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")

# =========================
# Initialize CNN
# =========================
model = Method_CNN('CNN', '')
model.acc_history = []  # record accuracy per epoch

# =========================
# Fit CNN
# =========================
metrics_dict = model.fit(X_train, y_train, X_test, y_test, epochs=30, batch_size=32)

# =========================
# Plot loss curve
# =========================
plt.figure()
plt.plot(model.loss_history, label='Training Loss')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("CNN Training Loss Curve")
plt.grid()
plt.legend()
plt.show()

# =========================
# Plot accuracy curve
# =========================
if hasattr(model, 'acc_history') and model.acc_history:
    plt.figure()
    plt.plot(model.acc_history, label='Training Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Training Accuracy Curve")
    plt.grid()
    plt.legend()
    plt.show()
else:
    print("No accuracy history recorded in the model.")

# =========================
# Compute final metrics
# =========================
true_y = metrics_dict['true_y']
pred_y = metrics_dict['pred_y']

accuracy = accuracy_score(true_y, pred_y)
f1 = f1_score(true_y, pred_y, average='macro')
precision = precision_score(true_y, pred_y, average='macro', zero_division=0)
recall = recall_score(true_y, pred_y, average='macro', zero_division=0)

print("\nCNN Test Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# =========================
# Plot CNN architecture (minimal change)
# =========================
# Build layer names dynamically after fit()
layers = []

# Conv layers
layers.append(f"Conv1\n{model.conv1.out_channels}x{X_train.shape[2]}x{X_train.shape[3]}")
if hasattr(model, 'bn1'):
    layers.append(f"BN1\n{model.bn1.num_features}")
layers.append(f"Conv2\n{model.conv2.out_channels}x{X_train.shape[2]//2}x{X_train.shape[3]//2}")
if hasattr(model, 'bn2'):
    layers.append(f"BN2\n{model.bn2.num_features}")
layers.append(f"Conv3\n{model.conv3.out_channels}x{X_train.shape[2]//4}x{X_train.shape[3]//4}")
if hasattr(model, 'bn3'):
    layers.append(f"BN3\n{model.bn3.num_features}")

# Fully connected layers
layers.append(f"FC1\n{model.fc1.out_features}")
layers.append(f"FC2\n{model.fc2.out_features}")

y_pos = list(range(len(layers)))

plt.figure(figsize=(6, 4))
plt.scatter([1]*len(y_pos), y_pos)

for i, layer in enumerate(layers):
    plt.text(1, y_pos[i], layer, ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle="round", facecolor="lightblue"))

for i in range(len(y_pos)-1):
    plt.plot([1,1], [y_pos[i], y_pos[i+1]], 'k-')

plt.title("CNN Architecture")
plt.axis('off')
plt.show()