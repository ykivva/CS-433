import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from proj1_helpers import *
from preprocessing import *
from nn_model import *


DATA_TRAIN_PATH = os.path.join(os.getcwd(), '../data/train.csv')
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

#Rename label -1 to 0; -999 in training set changes to nan
y[y==-1] = 0
y = y.astype(np.int)
tX[tX==-999] = np.nan

#Data balance
pos_zero = np.argwhere(y==0).squeeze()
num_delete = (y==0).sum() - (y==1).sum()
pos_delete = np.random.choice(pos_zero, replace=False, size=num_delete)
y_train = np.delete(y, pos_delete)
x_train = np.delete(tX, pos_delete, axis=0)

#Shuffle of training set
shuffle = np.arange(y_train.shape[0])
np.random.shuffle(shuffle)
y_train = y_train[shuffle]
x_train = x_train[shuffle]

#Set parameters
accuracy = 100
use_transformation = True
handling_outlier = 'fill_mean'
transform_inplace = False
max_degree = 2
lr = 1
lambda_ = 0
batch_size = 64
epochs = 25
momentum = 0.9
verbose = 1

#Make preprocessing
preprocessing = Preprocessing(use_transformations=use_transformation,
                              handling_outliers=handling_outlier,
                              max_degree=max_degree)
tX_preprocessed = preprocessing.preprocess(data_=tX, transform_inplace=transform_inplace)

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

x_test = preprocessing.preprocess(data_=tX_test, transform_inplace=transform_inplace)

OUTPUT_PATH = 'prediction.csv' # TODO: fill in desired name of output file for submission
y_pred = model.predict(x_test)
res = y_pred>0
res = res.squeeze()
pred = -np.ones(res.shape)
pred[res] = 1
create_csv_submission(ids_test, pred, OUTPUT_PATH)
print("Prediction created! Ready to submit!")