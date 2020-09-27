import numpy as np


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def softmax(output):
    max_deg = np.max(output, axis=1, keepdims=True)
    output_shift = output - max_deg
    return  np.exp(output_shift) / np.sum(output_shift, axis=1, keepdims=True)


def forward_backward(y, tx, w, lambda_=0):
    '''Compute gradient of Least Squares loss function

    Args:
        y (np.array): true predictions
        tx (np.array): data features
        w (np.array): model weights

    Returns:
        grads (np.array): gradient with respect to model weights
    '''
    epsilon = 10**(-5)
    x = np.hstack((np.ones(tx.shape[0], 1), tx))
    h = x @ w
    output = sigmoid(h)
    
    loss = -np.dot(y,  np.log(output)) - np.dot(1 - y, np.log(1 - output)) + lambda_ * np.linalg.norm(w[1:, ...])**2

    doutput = y / (output + epsilon) + (1 - y) / (1 - output + epsilon)
    dh = doutput * output * (1 - output)
    dw = x.T @ dh
    grads = dw
    grads[1:, ...] += 2 * lambda_ * w[1:, ...] 

    return (loss, grads)


def logistic_regression(y, tx, initial_w, max_iter, gamma):
    '''Stochastic Gradient Descent for logistic regression method
    
    Args:
        y (np.array): true predictions
        tx (np.array): data features
        initial_w (np.array): initial model weights
        max_iters (int): number of iterations
        gamma (float): learning rate parameter
    Returns:
        w (np.array): model weigths after the last iteration
        loss (float): loss value after the last iteration on corresponding batch
    '''
    x = np.hstack((np.ones(tx.shape[0], 1), tx))
    w = initial_w

    for iter_ in range(max_iter):
        loss, grads = forward_backward(y, x, w)
        w -= gamma * grads

    return (w, loss)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iter, gamma):
    '''Stochastic Gradient Descent for logistic regression method using batch size = 1
    
    Args:
        y (np.array): true predictions
        tx (np.array): data features
        labbda_ (float): coefficient for regression
        initial_w (np.array): initial model weights
        max_iters (int): number of iterations
        gamma (float): learning rate parameter
    Returns:
        w (np.array): model weigths after the last iteration
        loss (float): loss value after the last iteration on corresponding batch
    '''
    x = np.hstack((np.ones(tx.shape[0], 1), tx))
    w = initial_w

    for iter_ in range(max_iter):
        sample_num = np.random.randint(tx.shape[0])
        x_sgd = x[sample_num:sample_num+1]
        loss, grads = forward_backward(y, x_sgd, w, lambda_)
        w -= gamma * grads

    return (w, loss)