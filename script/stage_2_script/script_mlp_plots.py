# script_mlp_plots.py
import matplotlib.pyplot as plt
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Dataset_Loader import Dataset_Loader

# Plotting MLP architecture
layers = ["Input\n784", "FC1\n512", "FC2\n256", "FC3\n128", "Output\n10"]
y = [0, 1, 2, 3, 4]

plt.figure(figsize=(6, 4))
plt.scatter([1] * len(y), y)

for i, layer in enumerate(layers):
    plt.text(1, y[i], layer, ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle="round", facecolor="lightblue"))

for i in range(len(y) - 1):
    plt.plot([1, 1], [y[i], y[i + 1]], 'k-')

plt.title("MLP Architecture")
plt.axis('off')
plt.show()

# Function to plot loss
def plot_loss(model):
    plt.figure()
    plt.plot(model.loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid()
    plt.show()

# Function to plot accuracy
def plot_accuracy(model):
    plt.figure()
    plt.plot(model.acc_history)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Curve")
    plt.grid()
    plt.show()

# Data Loading section
data_obj = Dataset_Loader('MNIST', '')
data_obj.dataset_source_folder_path = '../../data/stage_2_data/'  # Set the correct path to the data folder
data_obj.train_file_name = 'train.csv'
data_obj.test_file_name = 'test.csv'

# Load the data (returns a dictionary)
loaded_data = data_obj.load()

# Initialize and run the model
model = Method_MLP("MLP", "baseline")
model.data = {
    'train': {
        'X': loaded_data['X_train'],
        'y': loaded_data['y_train']
    },
    'test': {
        'X': loaded_data['X_test'],
        'y': loaded_data['y_test']
    }
}

# Run the model and get metrics
metrics, pred_y = model.run()

# Plot loss and accuracy after training
plot_loss(model)
plot_accuracy(model)