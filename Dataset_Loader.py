from code.base_class.dataset import dataset
import pandas as pd

class Dataset_Loader(dataset):

    def load(self):
        print("loading data...")

        train_path = self.dataset_source_folder_path + "train.csv"
        test_path = self.dataset_source_folder_path + "test.csv"

        train_data = pd.read_csv(train_path, header=None)
        test_data = pd.read_csv(test_path, header=None)

        # FIRST column = label
        y_train = train_data.iloc[:, 0].values
        X_train = train_data.iloc[:, 1:].values

        y_test = test_data.iloc[:, 0].values
        X_test = test_data.iloc[:, 1:].values

        return {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test}
        }