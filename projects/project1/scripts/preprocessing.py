import numpy as np


def standardize(data):
    mu = np.nanmean(data, axis=0)
    sigma = np.nanstd(data, axis=0)
    return (data-mu)/sigma


def replace_outliers_by_nan(data, cols_with_outliers, outlier=-999.):
    for col in cols_with_outliers:
        data[data[:,col] == outlier,col] = np.nan


def get_transformation_dict():
    #get dict in 'column: transformation' format
    transformations = {
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
    return transformations


def preprocessing(data):
    
    data_preprocessed = data.copy()
    #processing outliers
    cols_with_outliers = [0, 4, 5, 6, 12, 23, 24, 25, 26, 27, 28]
    outlier = -999.
    replace_outliers_by_nan(data_preprocessed, cols_with_outliers, outlier)
    
    #feature transformation
    transformations = get_transformation_dict()
    for col in transformations.keys():
        data_preprocessed[:,col] = transformations[col](data_preprocessed[:,col])
    
    #convert categorical feature in one-hot format
    categorical_col = 22
    categories_num = 4
    data_numerical = np.concatenate([data_preprocessed[:,:categorical_col],
                                     data_preprocessed[:,categorical_col+1:]], axis=1)
    data_categorical = np.zeros([data_preprocessed.shape[0], categories_num])
    for i in range(data_preprocessed.shape[0]):
        val = data_preprocessed[i,categorical_col]
        #check for fidelity
        if (0 <= val and val < categories_num):
            data_categorical[i,int(val)] = 1
    data_preprocessed = np.concatenate([data_numerical, data_categorical], axis=1)
    
    #normalization of numerical features
    data_preprocessed[:,:29] = standardize(data_preprocessed[:,:29])
    return data_preprocessed