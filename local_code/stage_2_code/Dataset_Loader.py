'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset

class Dataset_Loader(dataset):
    dataset_source_folder_path = None
    train_file_name = None
    test_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')

        # Load training data
        X_train = []
        y_train = []
        with open(self.dataset_source_folder_path + self.train_file_name, 'r') as f:
            for line in f:
                line = line.strip()
                elements = [int(i) for i in line.split(',')]
                y_train.append(elements[0])
                X_train.append(elements[1:])

        # Load testing data
        X_test = []
        y_test = []
        with open(self.dataset_source_folder_path + self.test_file_name, 'r') as f:
            for line in f:
                line = line.strip()
                elements = [int(i) for i in line.split(',')]
                y_test.append(elements[0])
                X_test.append(elements[1:])

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }
