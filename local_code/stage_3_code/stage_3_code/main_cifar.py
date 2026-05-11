# ---- main_cifar.py ----

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN import Method_CNN
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Train_Test_Split import Setting_Train_Test_Split

dataset = Dataset_Loader(
    'CIFAR',
    'Colored Object Dataset'
)

dataset.dataset_source_folder_path = './datasets/'
dataset.dataset_source_file_name = 'cifar.pkl'

method = Method_CNN(
    'CNN_CIFAR',
    'CNN for Object Recognition',
    image_height=32,
    image_width=32,
    input_channels=3,
    num_classes=10
)

result = Result_Saver()

result.result_destination_folder_path = './results/'
result.result_destination_file_name = 'cnn_cifar_result'

setting = Setting_Train_Test_Split()

setting.dataset = dataset
setting.method = method
setting.result = result

metrics = setting.load_run_save_evaluate()

print(metrics)