import numpy as np


def compute_least_squares_loss(y, tx, w):
    '''Compute Least Squares loss

    Args:
        y (np.array): true predictions
        tx (np.array): data features
        w (np.array): model weights
    
    Returns:
        loss (float):
    '''
    n = y.shape[0]
    errors = y - np.dot(tx,w)
    loss = 1./(2*n) * np.dot(errors.T, errors)
    return loss


def compute_least_squares_gradient(y, tx, w):
    '''Compute gradient of Least Squares loss function

    Args:
        y (np.array): true predictions
        tx (np.array): data features
        w (np.array): model weights

    Returns:
        grad (np.array): gradient with respect to model weights
    '''
    n = y.shape[0]
    error = y - np.dot(tx,w)
    grad = -1./n * np.dot(tx.T, error)
    return grad


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    '''
    Gradient Descent for Least Squares method

    Args:
        y (np.array): true predictions
        tx (np.array): data features
        initial_w (np.array) initial model weights
        max_iters (int): number of iterations
        gamma (float): learning rate parameter

    Returns:
        w (np.array): model weigths after the last iteration
        loss (float): loss value after the last iteration
    '''
    w = initial_w
    for i in range(max_iters):
        grad = compute_least_squares_gradient(y, tx, w)
        w = w - gamma*grad
    loss = compute_least_squares_loss(y, tx, w)
    return (w, loss)