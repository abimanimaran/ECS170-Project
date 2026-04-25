from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_SVM import Method_SVM
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np

# ---- Support Vector Machine script ----
if 1:
    # ---- parameter section -------------------------------
    c = 1.0
    np.random.seed(2)
    # ------------------------------------------------------

    # ---- object initialization section ---------------
    data_obj = Dataset_Loader('MNIST', '')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    data_obj.train_file_name = 'train.csv'
    data_obj.test_file_name = 'test.csv'

    method_obj = Method_SVM('support vector machine', '')
    method_obj.c = c

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/SVM_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Split('train test split', '')
    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    # Call prepare with 4 positional arguments
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()

    # Load data, run method, save results
    metrics = setting_obj.load_run_save_evaluate()

    # Pass predictions to evaluator
    evaluate_obj.data = metrics
    eval_results = evaluate_obj.evaluate()

    print('************ Overall Performance ************')
    print('SVM Evaluation Metrics:', eval_results)
    print('************ Finish ************')
    # ------------------------------------------------------    # ------------------------------------------------------