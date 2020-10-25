# CS-433
This repository consists of implementation for project 1 and project 2 of [**EPFL CS-433**](https://edu.epfl.ch/coursebook/en/machine-learning-CS-433) course.

# Introduction
In the folder ```project1``` you can find all the code for [**EPFL Machine Learning Higgs**](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs) competition (2020). The goal of the project was to implement basic algorithms like linear and logistic regressions and their regularized versions, and use them for the competition. Also we have to make data investigation and preprocessing, so we can achieve much better results. 

On this competition our best result was achieved with *accuracy=0.820* and *F1-score=0.693*.

**`Team`**: **SomeGuys**

**`TeamMembers`**: **Yaroslav Kivva**, **Denys Pushkin**, **Odysseas Drosis**

# Quickstart
Before running the code make sure that you download train and test datasets to the directory ```/projects/project1/data/```.
All experimentation with training and making prediction was made in ```projects/project1/scripts/```

To reproduce our best results please follow the instructions below:
1. Download train and test datasets to the ```/projects/project1/data/```
2. ```cd project1```
3. ```cd script```
4. ```python run.py```

## Implementation

### **`proj1_helpers.py`**
Helper function to load train and test data and to generate output file with predictions for test dataset.

### **`implementations.py`**
There you can find 6 implemented basic algorithms for binary classification and regression.

- ```least_squares_GD(y, tx, initial_w, max_iters, gamma)```:  Linear regression using gradient descent
- ```least_squares_SGD(y, tx, initial_w, max_iters, gamma)```: Linear regression using stochastic gradient descent
- ```least_squares(y, tx)```: Least squares regression using normal equations
- ```ridge_regression(y, tx, lambda_)```: Ridge regression using normal equations
- ```logistic_regression(y, tx, initial w, max_iters, gamma)```: Logistic regression using gradient descent or SGD
- ```reg_logistic_regression(y, tx, lambda_, initial w, max iters, gamma)```: Regularized logistic regression using gradient descent or SGD

### **`nn_model.py`**
Module in which implemented class ```NNModel``` for training specified model and store its weights.
- ```__init__(self,  features_in)```: creates an instance of class, input of which has *features_in* features
- ```add_layer(self, units, activation=None)```: regular denesely-connected NN layer
  - ```units``` - number of features in the layer
  - ```activation``` - if `sigmoid`, after linear transformation sigmoid function applied to its output; if *None* any function applies to the output of linear transformation
- ```train(self, x, y, lr=0.1, lambda_=0, batch_size=None, epochs=1, verbose=0, loss_fun='l2', momentum=0)```: train the specified model
  - ```x```: train features
  - ```y```: train labels
  - ```lr```: learning rate for the gradient descent
  - ```batch_size```: batch size for the training
  - ```epochs```: number of epochs for the training
  - ```verbose```: if 1 it outputs loss after each epoch; if 0 nothing outputs
  - ```loss_fun```: *l2* - it computes *MSE* loss; *logistic_reg* - it computes sigmoid for the output and computes Negative Log Likelihood loss
  - ```momentum```: momentum parameter for gradient descent step with momentum
  - ```predict(self, x)```: returns output of the model

### **`preprocessing.py`**
Module in which implemented class ```Preprocessing``` to make transformation with training data, store values which was used for transformation and make the same transformations for test set. Main methods in the class:
- ```__init__(self, use_transformations=True, use_normalization=True, handling_outliers='fill_mean', max_degree=None)``` : initialize incstance of class which will store parameters and generated variables for preprocesing
  - ```use_transformations```: if *True* it will use transformation for some features specified in class so that their distribution will look more normal-like
  - ```handling_outliers```: if 'fill_mean', it replace all *nan/-999* values with their mean (after normalization *mean*=0); if 'remove', it just removes all columns with *nan/-999*
  - ```max_degree```: if `None` it will not use data augmentation, if *int* it will add to the features their degrees from 1 to max_degree(included)
- ```preprocess(self, data_, transform_inplace=True)```: performs data preprocessing to the *data_* and returns its copy
  - ```transform_inplace```: if True transformation will be done in place, otherwise added transformed data like new features

### **`project1.ipynb`**
Notebook for experimentations with training and predicting.
