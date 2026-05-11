'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
# local_code/stage_3_code/Dataset_Loader.py

import os
import numpy as np
import pickle

class Dataset_Loader:
    def __init__(self, dataset_name, dummy):
        self.dataset_name = dataset_name
        self.dataset_source_folder_path = ''
        self.dataset_source_file_name = ''

    def process_image(self, image):
        # Normalize image to [0,1] and add channel dim if grayscale
        image = np.array(image, dtype=np.float32) / 255.0
        if image.ndim == 2:
            # grayscale image
            image = image[np.newaxis, :, :]
        elif image.ndim == 3:
            # move channels first
            image = np.transpose(image, (2, 0, 1))
        return image

    def load(self):
        print("loading data...")

        file_path = os.path.join(self.dataset_source_folder_path, self.dataset_source_file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # single pickle file for CIFAR
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f, encoding='latin1')

        X_train, y_train = [], []
        X_test, y_test = [], []

        for sample in dataset['train']:
            X_train.append(self.process_image(sample['image']))
            if 'label' in sample:
                y_train.append(sample['label'])
            elif 'target' in sample:
                y_train.append(sample['target'])
            else:
                y_train.append(sample['class'])

        for sample in dataset['test']:
            X_test.append(self.process_image(sample['image']))
            if 'label' in sample:
                y_test.append(sample['label'])
            elif 'target' in sample:
                y_test.append(sample['target'])
            else:
                y_test.append(sample['class'])

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test  = np.array(X_test)
        y_test  = np.array(y_test)

        print("data loaded successfully!")
        print(f"X_train: {X_train.shape}")
        print(f"X_test: {X_test.shape}")

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }