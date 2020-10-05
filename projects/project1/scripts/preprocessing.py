import numpy as np
import matplotlib.pyplot as plt
from linear_regression import least_squares


class Preprocessing:
    
    
    def __init__(self, use_transformations=True, use_standartization=True, handling_outliers='predict'):
        
        self.categorical_col = 22
        self.categories_num = 4
        self.numerical_features = 29
        self.cols_with_outliers = [0, 4, 5, 6, 12, 23, 24, 25, 26, 27, 28]
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
        self.use_standartization = use_standartization
        self.handling_outliers = handling_outliers
        assert (self.handling_outliers in ['fill_mean', 'remove', 'predict'])
        
        #after basic precpocessing categorical column moved
        self.cols_with_NaNs = np.array([(x if x < self.categorical_col else x-1) for x in self.cols_with_outliers])
        total_cols = self.numerical_features + self.categories_num
        arr = np.array([True]*total_cols)
        arr[self.cols_with_NaNs] = False
        self.cols_without_NaNs = np.arange(total_cols)[arr]
        self.weights_for_missing_values_prediction = {}
        
        self.means = None
        self.stds = None
    
    
    def fit(self, train_data):
        
        data = train_data.copy() #do not want to change train_data
        self.replace_outliers_by_nan(data)
        if self.use_transformations:
            self.transform(data)
        data = self.convert_categories_to_one_hot(data)
        if self.use_standartization:
            self.means = np.nanmean(data[:,:self.numerical_features], axis=0)
            self.stds = np.nanstd(data[:,:self.numerical_features], axis=0)
            self.standardize(data)
        if self.handling_outliers == 'predict':
            self.evaluate_weights_for_missing_values_prediction(data)
    
    
    def preprocess(self, data_):

        data = data_.copy() #do not want to change data_
        self.replace_outliers_by_nan(data)
        if self.use_transformations:
            self.transform(data)
        data = self.convert_categories_to_one_hot(data)
        if self.use_standartization:
            self.standardize(data)
        if self.handling_outliers == 'remove':
            data = self.remove_rows_with_NaNs(data)
        elif self.handling_outliers == 'fill_mean':
            data = self.fill_NaNs_with_zeroes(data)
        elif self.handling_outliers == 'predict':
            print_nans_per_column(data)
            data = self.predict_NaNs(data)
            print_nans_per_column(data)
        else:
            raise ValueError('Value of handling_NaNs is not acceptable')
        return data
    
    
    def replace_outliers_by_nan(self, data):
        for col in self.cols_with_outliers:
            data[data[:,col] == self.outlier,col] = np.nan
    
    
    def standardize(self, data):
        if (self.means is None or self.stds is None):
            raise Exception('Cannot standardize data: need to fit train_data first')
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
    
    
    def evaluate_weights_for_missing_values_prediction(self, train_data):
        
        X = train_data[:, self.cols_without_NaNs]
        for col in self.cols_with_NaNs:
            Y_known = train_data[~np.isnan(train_data[:,col]), col]
            X_ = X[~np.isnan(train_data[:,col]), :]
            self.weights_for_missing_values_prediction[col], _ = least_squares(Y_known, X_)
    
    
    def remove_rows_with_NaNs(self, data):
        return data[:,self.cols_without_NaNs]

    
    def fill_NaNs_with_zeroes(self, data_):
        data = data_.copy() #do not want to change data_
        data[np.isnan(data)] = 0
        return data
    
    
    def predict_NaNs(self, data_):
        
        if len(self.weights_for_missing_values_prediction) == 0:
            raise Exception('Cannot predict NaN values: need to fit train_data first')
        data = data_.copy() #do not want to change data_
        X = data[:, self.cols_without_NaNs]
        X = np.concatenate([np.ones(shape=(X.shape[0], 1)), X], axis=1) #add bias term
        for col in self.cols_with_NaNs:
            data[np.isnan(data[:,col]), col] = np.dot(X[np.isnan(data[:,col]), :],
                                                      self.weights_for_missing_values_prediction[col])
        return data
        