# ---- main_mnist.py ----

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN import Method_CNN
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Train_Test_Split import Setting_Train_Test_Split

# =====================================================
# Dataset
# =====================================================

dataset = Dataset_Loader(
    'MNIST',
    'Handwritten Digits'
)

dataset.dataset_source_folder_path = './datasets/'
dataset.dataset_source_file_name = 'mnist.pkl'

# =====================================================
# CNN Method
# =====================================================

method = Method_CNN(
    'CNN_MNIST',
    'CNN for Digit Recognition',
    image_height=28,
    image_width=28,
    input_channels=1,
    num_classes=10
)

# =====================================================
# Result Saver
# =====================================================

result = Result_Saver()

result.result_destination_folder_path = './results/'
result.result_destination_file_name = 'cnn_mnist_result'

# =====================================================
# Setting
# =====================================================

setting = Setting_Train_Test_Split()

setting.dataset = dataset
setting.method = method
setting.result = result

# =====================================================
# Run
# =====================================================

metrics = setting.load_run_save_evaluate()

print(metrics)