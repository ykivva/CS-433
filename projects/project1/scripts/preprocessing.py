import numpy as np

class Preprocessing:
    '''
    class for preprocessing features for EPFL Machine Learning Higgs problem
    
    Args:
        k_categorical_col (int): index of categorical column (in initial dataset)
        k_numerical_features (int): number of numerical features (in initial dataset)
        k_cols_with_outliers (np.array): indexes of columns with outliers (in initial dataset)
        k_cols_with_NaNs (np.array): indexes of columns with NaNs after columns' shift in convert_categories_to_one_hot     function
        k_cols_without_NaNs (np.array): indexes of columns without NaNs after columns' shift in convert_categories_to_one_hot    function
        
    '''
    k_categorical_col = 22
    k_numerical_features = 29
    k_cols_with_outliers = np.array([0, 4, 5, 6, 12, 23, 24, 25, 26, 27, 28])
    k_cols_with_NaNs = None
    k_cols_without_NaNs = None

    def __init__(self, use_transformations=True, use_normalization=True,
                 handling_outliers='fill_mean', max_degree=None):
        '''
        Initialize instance of Preprocessing.
        
        Args:
            use_transformations (bool): whether to apply transformations for some features to make their distribution more normal-like
            use_normalization (bool): whether to apply features normalization
            handling_outliers (string): mode of habdling outliers. Can be 'fill_mean' or 'remove' (i. e. remove all columns that contain at least one outlier)
            max_degree (None or int): if int, max degree for polynomial features augmentation. If None, polynomial features augmentation will not be applied
        '''
        
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
        assert (self.handling_outliers in ['fill_mean', 'remove'])
        self.max_degree = max_degree
        
        #in convert_categories_to_one_hot functions categorical column moves to the end
        self.cols_with_NaNs = np.array([(x if x < self.categorical_col else x-1) for x in self.cols_with_outliers])
        self.k_cols_with_NaNs = self.cols_with_NaNs.copy()
        total_cols = self.numerical_features + self.categories_num
        arr = np.array([True]*total_cols)
        arr[self.cols_with_NaNs] = False
        self.cols_without_NaNs = np.arange(total_cols)[arr]
        self.k_cols_without_NaNs = self.cols_without_NaNs.copy()
        
        #statistics for initial features' normalization
        self.means = None
        self.stds = None
        #statistics for polynomially augmented features' normalization
        self.degrees_means = None
        self.degrees_stds = None
        
        self.is_fitted = False #whether train set was already fitted to derive some variable required for preprocessing
    
    def preprocess(self, data_, transform_inplace=True, pairwise=True, add_exp=False):
        '''
        Preprocess data_. The first fitted dataset will be used to derived means and stds for normalization of all further datasets fitted
        Args:
            data_ (np.array): dataset before preprocessing
            transform_inplace (bool): if use_transformations == True, defines whether to transform features inplace or to create additional columns for transformed features.
        Returns:
            data (np.array): dataset after preprocessing
        '''
        
        #set variables to initial state
        self.categorical_col = self.k_categorical_col
        self.numerical_features = self.k_numerical_features
        self.cols_with_outliers = self.k_cols_with_outliers.copy()
        self.cols_with_NaNs = self.k_cols_with_NaNs.copy()
        self.cols_without_NaNs = self.k_cols_without_NaNs.copy()

        data = data_.copy()
        self.replace_outliers_by_nan(data)
        if self.use_transformations:
            data = self.transform(data, transform_inplace)
        data = self.convert_categories_to_one_hot(data)
        
        if self.use_normalization:
            if not self.is_fitted:
                #fitting means and stds parameters
                self.means = np.nanmean(data[:,:self.numerical_features], axis=0)
                self.stds = np.nanstd(data[:,:self.numerical_features], axis=0)
            self.normalize(data)
        
        if self.max_degree != None:
            data = self.build_poly(data, self.max_degree, pairwise=pairwise, add_exp=add_exp)

        if self.handling_outliers == 'remove':
            data = self.remove_cols_with_NaNs(data)
        elif self.handling_outliers == 'fill_mean':
            data = self.fill_NaNs_with_zeroes(data) #after normalization, it is equavalent to filling NaNs with means
        else:
            raise ValueError('Value of handling_NaNs is not acceptable')
        
        if not self.is_fitted:
            self.is_fitted = True
        
        return data
    
    def replace_outliers_by_nan(self, data):
        data[data==self.outlier] = np.nan
    
    def normalize(self, data):
        data[:,:self.numerical_features] = (data[:,:self.numerical_features]-self.means)/self.stds
    
    def transform(self, data_, inplace):
        '''
        Returns a copy of data_ after applying to it feature transformations.
        Args:
            data_ (np.array): input dataset
            inplace (bool): whether to transform features inplace or to create additional columns for transformed features
        Returns:
            data (np.array): dataset after applying feature transformation
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
                self.cols_without_NaNs = np.append(np.zeros(1, dtype=np.int), self.cols_without_NaNs)
                shift += 1
        return data
    
    def convert_categories_to_one_hot(self, data):
        '''
        Convert categorical column to one-hot representation. Removes old categorical column and adds one-hot columns fot it in the end of the dataset
        '''
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
        data = data_.copy()
        data[np.isnan(data)] = 0
        return data    
    
    def build_poly(self, data_, max_degree, pairwise=True, add_exp=False):
        '''
        Apply polynomial augmentation and normalize augmented featerus
        Args:
            data_ (np.array): dataset to which polynomial augmentation will be applied
            max_degree (int): max degree for polynomial augmentation
        Returns:
            data (np.array): dataset after applying polynomial augmentation
        '''
        numerical_columns_without_NaNs = self.cols_without_NaNs[:-4] #columns to be augmented
        data = data_.copy()
        min_degree=2
        if not self.is_fitted:
            self.degrees_means = np.array([])
            self.degrees_stds = np.array([])
        
        if add_exp:
            for col in numerical_columns_without_NaNs:
                exp_data = np.exp(data_[:, numerical_columns_without_NaNs])
                data = np.hstack((exp_data, data))
                if not self.is_fitted:
                    self.degrees_means = np.concatenate([np.mean(exp_data, axis=0), self.degrees_means])
                    self.degrees_stds = np.concatenate([np.std(exp_data, axis=0), self.degrees_stds])

        
        if pairwise:
            for col in numerical_columns_without_NaNs:
                indx = numerical_columns_without_NaNs[numerical_columns_without_NaNs>col]
                pairwise_mult = data_[:, col:col+1] * data_[:, indx]
                data = np.hstack((pairwise_mult, data))
                if not self.is_fitted:
                    self.degrees_means = np.concatenate([np.mean(pairwise_mult, axis=0), self.degrees_means])
                    self.degrees_stds = np.concatenate([np.std(pairwise_mult, axis=0), self.degrees_stds])
        
        for deg in range(min_degree, max_degree+1):
            pol_data = data_[..., numerical_columns_without_NaNs]**deg
            data = np.hstack((pol_data, data))
            if not self.is_fitted:
                self.degrees_means = np.concatenate([np.mean(pol_data, axis=0), self.degrees_means])
                self.degrees_stds = np.concatenate([np.std(pol_data, axis=0), self.degrees_stds])
        
        num_initial_features = self.categories_num + self.numerical_features
        data[:,:-num_initial_features] = (data[:,:-num_initial_features] - self.degrees_means) / self.degrees_stds
        
        return data
