import numpy as np


def compute_least_squares_loss(y, tX, w):
    '''Compute Least Squares loss

    Args:
        y (np.array): true predictions
        tX (np.array): data features
        w (np.array): model weights
    
    Returns:
        loss (float):
    '''
    n = y.shape[0]
    errors = y - np.dot(tX,w)
    loss = 1./(2*n) * np.dot(errors.T, errors)
    return loss


def compute_least_squares_gradient(y, tX, w):
    '''Compute gradient of Least Squares loss function

    Args:
        y (np.array): true predictions
        tX (np.array): data features
        w (np.array): model weights

    Returns:
        grad (np.array): gradient with respect to model weights
    '''
    n = y.shape[0]
    error = y - np.dot(tX,w)
    grad = -1./n * np.dot(tX.T, error)
    return grad


def least_squares_GD(y, tX, initial_w, max_iters, lr):
    '''
    Gradient Descent for Least Squares method

    Args:
        y (np.array): true predictions
        tX (np.array): data features
        initial_w (np.array) initial model weights
        max_iters (int): number of iterations
        lr (float): learning rate parameter

    Returns:
        w (np.array): model weigths after the last iteration
        loss (float): loss value after the last iteration
    '''
    assert y.shape[0]==tX.shape[0], "First dimenstion of y doesn't match first dimentsion of tX"
    
    w = initial_w
    for i in range(max_iters):
        loss = compute_least_squares_loss(y, tX, w)
        grad = compute_least_squares_gradient(y, tX, w)
        w = w - lr*grad
    return (w, loss)