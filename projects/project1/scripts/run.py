import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from proj1_helpers import *
from preprocessing import *
from nn_model import *


DATA_TRAIN_PATH = os.path.join(os.getcwd(), '../data/train.csv')
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)


def convert_labels(y):
    y[y==-1] = 0
    y = y.astype(np.int)
    return y


def calculate_accuracy(y, x, model):
    y_pred = model.predict(x)
    y_pred = y_pred > 0
    y_pred = y_pred.squeeze()
    accuracy = (y_pred==y).mean()
    return accuracy


#Rename label -1 to 0
y = convert_labels(y)

#Set parameters
use_transformation = True
handling_outlier = 'fill_mean'
transform_inplace = False
max_degree = 2
pairwise=True
add_exp = True
lambda_ = 0
lr = 1
verbose = 1
batch_size = 64
epochs = 25
momentum = 0.9

#Initialize and make preprocessing
preprocessing = Preprocessing(use_transformations=use_transformation,
                              handling_outliers=handling_outlier,
                              max_degree=max_degree)        
tX_preprocessed = preprocessing.preprocess(data_=tX, transform_inplace=transform_inplace, pairwise=pairwise, add_exp=add_exp)

#Initialization and training the model
model = NNModel(tX_preprocessed.shape[1])
model.add_layer(1)
model.train(tX_preprocessed, y,
            lr=lr, lambda_=lambda_,
            batch_size=batch_size,
            epochs=epochs, verbose=verbose,
            loss_fun='logistic_reg', momentum=momentum)


y_pred = model.predict(tX_preprocessed)
y_pred = y_pred > 0
y_pred = y_pred.squeeze()
print('\nAccuracy on the training set:', (y_pred==y).mean())

DATA_TEST_PATH = DATA_TRAIN_PATH = os.path.join(os.getcwd(), '../data/test.csv')
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

x_test = preprocessing.preprocess(data_=tX_test, transform_inplace=transform_inplace, pairwise=pairwise, add_exp=add_exp)

OUTPUT_PATH = 'prediction.csv' # TODO: fill in desired name of output file for submission
y_pred = model.predict(x_test)
res = y_pred>0
res = res.squeeze()
pred = -np.ones(res.shape)
pred[res] = 1
create_csv_submission(ids_test, pred, OUTPUT_PATH)
print("Prediction created! Ready to submit!")