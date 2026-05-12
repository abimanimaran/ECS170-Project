# ---- main_orl.py ----

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN import Method_CNN
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Train_Test_Split import Setting_Train_Test_Split

dataset = Dataset_Loader(
    'ORL',
    'Face Dataset'
)

dataset.dataset_source_folder_path = './datasets/'
dataset.dataset_source_file_name = 'orl.pkl'

method = Method_CNN(
    'CNN_ORL',
    'CNN for Face Recognition',
    image_height=112,
    image_width=92,
    input_channels=1,
    num_classes=40
)

result = Result_Saver()

result.result_destination_folder_path = './results/'
result.result_destination_file_name = 'cnn_orl_result'

setting = Setting_Train_Test_Split()

setting.dataset = dataset
setting.method = method
setting.result = result

metrics = setting.load_run_save_evaluate()

print(metrics)