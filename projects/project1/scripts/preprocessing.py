import numpy as np

class Preprocessing:

    k_categorical_col = 22
    k_numerical_features = 29
    k_cols_with_outliers = np.array([0, 4, 5, 6, 12, 23, 24, 25, 26, 27, 28])
    k_cols_with_NaNs = None
    k_cols_without_NaNs = None

    def __init__(self, use_transformations=True, use_normalization=True,
                 handling_outliers='fill_mean', max_degree=None):
        
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
        self.max_degree = max_degree
        
        #after basic precpocessing categorical column moved
        self.cols_with_NaNs = np.array([(x if x < self.categorical_col else x-1) for x in self.cols_with_outliers])
        self.k_cols_with_NaNs = self.cols_with_NaNs
        total_cols = self.numerical_features + self.categories_num
        arr = np.array([True]*total_cols)
        arr[self.cols_with_NaNs] = False
        self.cols_without_NaNs = np.arange(total_cols)[arr]
        self.k_cols_without_NaNs = self.cols_without_NaNs
        
        #statistics for initial features
        self.means = None
        self.stds = None
        #statistics for polynomially augmented features, if applicable 
        self.degrees_means = None
        self.degrees_stds = None
        
        self.is_fitted = False #whether train set was already fitted to derive mean, stds and NaNprediction model parameters
    
    def preprocess(self, data_, transform_inplace=True):
        if self.is_fitted:
            self.categorical_col = self.k_categorical_col
            self.numerical_features = self.k_numerical_features
            self.cols_with_outliers = self.k_cols_with_outliers
            self.cols_with_NaNs = self.k_cols_with_NaNs
            self.cols_without_NaNs = self.k_cols_without_NaNs

        data = data_.copy() #do not want to change data_
        self.replace_outliers_by_nan(data)
        if self.use_transformations:
            data = self.transform(data, inplace=transform_inplace)
        data = self.convert_categories_to_one_hot(data)
        
        if self.use_normalization:
            if not self.is_fitted:
                #fitting means and stds parameters
                self.means = np.nanmean(data[:,:self.numerical_features], axis=0)
                self.stds = np.nanstd(data[:,:self.numerical_features], axis=0)
            self.normalize(data)
        
        if self.max_degree != None:
            data = self.build_poly(data, self.max_degree)

        if self.handling_outliers == 'remove':
            data = self.remove_cols_with_NaNs(data)
        elif self.handling_outliers == 'fill_mean':
            data = self.fill_NaNs_with_zeroes(data)
        else:
            raise ValueError('Value of handling_NaNs is not acceptable')
        
        if not self.is_fitted:
            self.is_fitted = True
        
        return data
    
    def replace_outliers_by_nan(self, data):
        data[data==self.outlier] = np.nan
    
    def normalize(self, data):
        data[:,:self.numerical_features] = (data[:,:self.numerical_features]-self.means)/self.stds
    
    def transform(self, data_, inplace=True):
        '''
        transforming some features inplace (i. e. without saving features before transformations)
        '''
        data = data_.copy()
        if inplace:
            for col in self.transformations.keys():
                data[:,col] = self.transformations[col](data[:,col])
        else:
            shift = 0
            for col in self.transformations.keys():
                data = np.hstack((self.transformations[col](data[:,col+shift:col+shift+1]), data))
                self.cols_with_NaNs += 1
                self.cols_with_outliers += 1
                self.cols_without_NaNs += 1
                self.categorical_col += 1
                self.numerical_features += 1
                self.cols_without_NaNs = np.append(self.cols_without_NaNs, 0)
                shift += 1
        return data
    
    def convert_categories_to_one_hot(self, data):
        
        data_numerical = np.concatenate([data[:,:self.categorical_col],
                                     data[:,self.categorical_col+1:]], axis=1)
        data_categorical = np.zeros([data.shape[0], self.categories_num])
        for i in range(data.shape[0]):
            val = data[i, self.categorical_col]
            #check for fidelity
            if (0 <= val and val < self.categories_num):
                data_categorical[i, int(val)] = 1
        data = np.concatenate([data_numerical, data_categorical], axis=1)
        return data

    @staticmethod
    def get_cols_without_NaN(data):
        mask = np.any(np.isnan(data), axis=0)
        return data[..., ~mask] 
        
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
    
    def build_poly(self, data_, max_degree):
        numerical_columns_without_NaNs = self.cols_without_NaNs[:-4] #columns to be augmented
        data = data_.copy()
        if not self.is_fitted:
            degrees_means = np.array([])
            degrees_stds = np.array([])
        
        for deg in range(2, max_degree+1):
            pol_data = data_[..., numerical_columns_without_NaNs]**deg
            data = np.hstack((pol_data, data))
            if not self.is_fitted:
                degrees_means = np.concatenate([np.mean(pol_data, axis=0), degrees_means])
                degrees_stds = np.concatenate([np.std(pol_data, axis=0), degrees_stds])
        
        num_initial_features = self.categories_num + self.numerical_features
        data[:,:-num_initial_features] = (data[:,:-num_initial_features] - degrees_means) / degrees_stds
        
        return data
