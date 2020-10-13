import numpy as np
import matplotlib.pyplot as plt
from nn_model import NNModel


class Preprocessing:
        
    def __init__(self, use_transformations=True, use_normalization=True, handling_outliers='predict'):
        
        self.categorical_col = 22
        self.categories_num = 4
        self.numerical_features = 29
        self.cols_with_outliers = np.array([0, 4, 5, 6, 12, 23, 24, 25, 26, 27, 28])
        self.outlier = -999.
        self.transformations = {
            0: np.log,
            1: np.sqrt,
            2: np.log,
            3: lambda x: np.log(1+x),
            4: np.sqrt,
            5: np.log,
            8: lambda x: np.log(1+x),
            9: np.log,
            10: np.log,
            13: lambda x: np.log(x-19),
            16: lambda x: np.log(x-25),
            19: np.log,
            21: np.log,
            23: lambda x: np.log(x-29),
            26: lambda x: np.log(x-29)
        }
        self.use_transformations = use_transformations
        self.use_normalization = use_normalization
        self.handling_outliers = handling_outliers
        assert (self.handling_outliers in ['fill_mean', 'remove', 'predict'])
        
        #after basic precpocessing categorical column moved
        self.cols_with_NaNs = np.array([(x if x < self.categorical_col else x-1) for x in self.cols_with_outliers])
        total_cols = self.numerical_features + self.categories_num
        arr = np.array([True]*total_cols)
        arr[self.cols_with_NaNs] = False
        self.cols_without_NaNs = np.arange(total_cols)[arr]
        
        self.means = None
        self.stds = None
    
    def preprocess(self, data_, lr=0.03, lambda_=1, batch_size=32, epochs=100, degrees=np.arange(2, 4), train=True, models={}):

        print("Preprocesing started!\n")
        data = data_.copy() #do not want to change data_
        self.replace_outliers_by_nan(data)
        if self.use_transformations:
            self.transform(data)
        data = self.convert_categories_to_one_hot(data)
        
        data = self.build_poly(data, degrees, self.cols_without_NaNs)
        if self.use_normalization:
            self.means = np.nanmean(data[:,:self.numerical_features], axis=0)
            self.stds = np.nanstd(data[:,:self.numerical_features], axis=0)
            self.normalize(data)

        if self.handling_outliers == 'remove':
            data = self.remove_cols_with_NaNs(data)
        elif self.handling_outliers == 'fill_mean':
            data = self.fill_NaNs_with_zeroes(data)
        elif self.handling_outliers == 'predict':
            data, models = self.predict_Nans(data, lr=lr, lambda_=lambda_, batch_size=batch_size, epochs=epochs, train=train, models=models)
        else:
            raise ValueError('Value of handling_NaNs is not acceptable')

        print("Preprocessing ended\n")
        return data, models
    
    def replace_outliers_by_nan(self, data):
        data[data==self.outlier] = np.nan
    
    def normalize(self, data):
        if (self.means is None or self.stds is None):
            raise Exception('Cannot normalize data: need to fit train_data first')
        data[:,:self.numerical_features] = (data[:,:self.numerical_features]-self.means)/self.stds
    
    def transform(self, data):
        for col in self.transformations.keys():
            data[:,col] = self.transformations[col](data[:,col])
    
    def convert_categories_to_one_hot(self, data):
        
        data_numerical = np.concatenate([data[:,:self.categorical_col],
                                     data[:,self.categorical_col+1:]], axis=1)
        data_categorical = np.zeros([data.shape[0], self.categories_num])
        for i in range(data.shape[0]):
            val = data[i, self.categorical_col]
            #check for fidelity
            if (0 <= val and val < self.categories_num):
                data_categorical[i,int(val)] = 1
        data = np.concatenate([data_numerical, data_categorical], axis=1)
        return data

    @staticmethod
    def get_cols_without_NaN(data):
        mask = np.any(np.isnan(data), axis=0)
        return data[..., ~mask] 

    def predict_Nans(self, data_, lr=0.03, lambda_=1, batch_size=32, epochs=100, train=True, models={}):
        data = data_.copy()
        models = models
        for j in range(data.shape[1]):
            if not np.any(np.isnan(data[:, j])):
                continue

            features = Preprocessing.get_cols_without_NaN(data)
            mask_rows_with_nan = np.isnan(data[:, j])

            if train:
                train_features = features[~mask_rows_with_nan, ...]
                train_labels = data[~mask_rows_with_nan, j]

                models[j] = NNModel(train_features.shape[1])
                model = models[j]
                model.add_layer(1)

                model.train(train_features, train_labels, lr = lr, lambda_=lambda_, batch_size=batch_size, epochs=epochs, verbose=1, loss_fun='l2')
            else:
                model = models[j]

            data[mask_rows_with_nan, j:j+1] = model.predict(features[mask_rows_with_nan, ...])
        
        return data, models
        
    def remove_rows_with_NaNs(self, data):
        mask = np.isnan(data)
        mask_rows_with_nan = np.any(mask, axis=1)
        return data[mask_rows_with_nan, ...]

    def remove_cols_with_NaNs(self, data):
        mask = np.isnan(data)
        mask_cols_with_nan = np.any(mask, axis=0)
        return data[..., ~mask_cols_with_nan]

    def fill_NaNs_with_zeroes(self, data_):
        data = data_.copy() #do not want to change data_
        data[np.isnan(data)] = 0
        return data    
    
    def build_poly(self, data_, degrees, columns):
        data = data_.copy()

        for deg in degrees:
            pol_data = data[..., columns]**deg
            data = np.hstack((pol_data, data))
        
        num_col_added = len(degrees) * len(columns)
        self.numerical_features += num_col_added
        self.cols_without_NaNs += num_col_added
        self.cols_with_NaNs += num_col_added
        self.cols_without_NaNs = np.hstack((self.cols_without_NaNs, np.arange(num_col_added)))
        self.cols_with_outliers += num_col_added
        return data