import numpy as np


def sigmoid(t):
    """apply the sigmoid function on t."""
    
    if np.isscalar(t):
        if t >= 0:
            res = 1 / (1 + np.exp(-t))
        else:
            res = np.exp(t) / (np.exp(t) + 1)
    else:
        res = t.copy()
        res[t>=0] = 1 / (1 + np.exp(-t[t>=0]))
        res[t<0] = np.exp(t[t<0]) / (np.exp(t[t<0]) + 1)
    return res

def calculate_loss(y, tx, w, lambda_=0):
    """compute the loss: negative log likelihood."""
    
    loss = np.sum(np.log(1+np.exp(tx @ w)) - y * (tx @ w)) + lambda_/2 * np.dot(w.T,w)
    return loss

def calculate_gradient(y, tx, w, lambda_=0):
    """compute the gradient of loss."""
    
    grad = tx.T @ (sigmoid(tx @ w) - y) + lambda_ * w
    return grad


def logistic_regression(y, tx, initial_w, max_iter, lr):
    '''Gradient Descent for logistic regression method
    
    Args:
        y (np.array): true predictions
        tx (np.array): data features
        initial_w (np.array): initial model weights
        max_iters (int): number of iterations
        lr (float): learning rate parameter
    Returns:
        w (np.array): model weigths after the last iteration
        loss (float): loss value after the last iteration on corresponding batch
    '''
    
    w = initial_w.copy()
    
    for iter_ in range(max_iter):
        loss = calculate_loss(y, tx, w)
        grad = calculate_gradient(y, tx, w)        
        w -= lr * grad

    return (w, loss)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iter, lr):
    '''Gradient Descent for regularized logistic regression
    
    Args:
        y (np.array): true predictions
        tX (np.array): data features
        lambda_ (float): coefficient for regression
        initial_w (np.array): initial model weights
        max_iters (int): number of iterations
        lr (float): learning rate parameter
    Returns:
        w (np.array): model weigths after the last iteration
        loss (float): loss value after the last iteration on corresponding batch
    '''
    
    w = initial_w.copy()
    
    for iter_ in range(max_iter):
        loss = calculate_loss(y, tx, w, lambda_)
        grad = calculate_gradient(y, tx, w, lambda_)        
        w -= lr * grad

    return (w, loss)


def logistic_pred(tx, w):
    y_prob = sigmoid(tx @ w)
    y_pred = y_prob > 0.5
    return y_pred.squeeze()