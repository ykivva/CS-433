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


def least_squares_GD(y, tX, initial_w, max_iters, gamma):
    '''
    Gradient Descent for Least Squares method

    Args:
        y (np.array): true predictions
        tX (np.array): data features
        initial_w (np.array) initial model weights
        max_iters (int): number of iterations
        gamma (float): learning rate parameter

    Returns:
        w (np.array): model weigths after the last iteration
        loss (float): loss value after the last iteration
    '''
    assert y.shape[0]==tX.shape[0], "First dimenstion of y doesn't match first dimentsion of tX"
    
    w = initial_w
    for i in range(max_iters):
        loss = compute_least_squares_loss(y, tX, w)
        grad = compute_least_squares_gradient(y, tX, w)
        w = w - gamma*grad
    return (w, loss)


def least_squares_SGD(y, tX, initial_w, max_iters, gamma):
    '''Stochastic Gradient Descent for Least Squares method using batch size = 1
    
    Args:
        y (np.array): true predictions
        tX (np.array): data features
        initial_w (np.array): initial model weights
        max_iters (int): number of iterations
        gamma (float): learning rate parameter
    Returns:
        w (np.array): model weigths after the last iteration
        loss (float): loss value after the last iteration on corresponding batch
    '''
    assert y.shape[0]==tX.shape[0], "First dimenstion of y doesn't match first dimentsion of tX"
    
    w = initial_w
    batch_numbers = np.random.randint(0, y.shape[0], size=max_iters)
    for i in batch_numbers:
        loss = compute_least_squares_loss(y[i:i+1], tX[i:i+1], w)
        grad = compute_least_squares_gradient(y[i:i+1], tX[i:i+1], w)
        w = w - gamma*grad
    return (w, loss)


def least_squares(y, tX):
    """Finds optimal solution for linear regression using normal equations

    Args:
        y (nd.array): true predictions
        tX (nd.array): data features

    Returns:
        Tuple (nd.array, float), where first is parameters and second loss value
    """
    assert y.shape[0]==tX.shape[0], "First dimenstion of y doesn't match first dimentsion of tX"
    
    n = tX.shape[0]
    w = np.linalg.pinv(tX) @ y
    loss = 1/(2*n) * np.linalg.norm(tX @ w - y)**2

    return (w, loss)


def ridge_regression(y, tX, lambda_):
    """Finds optimal solution for ridge regression using normal equations

    Args:
        y (nd.array): true predictions
        tX (nd.array): data features
        labbda_ (float): coefficient for regression

    Returns:
        Tuple (nd.array, float), where first is parameters and second loss value
    """
    assert y.shape[0]==tX.shape[0], "First dimenstion of y doesn't match first dimentsion of tX"

    n = tX.shape[0]
    lambda_hat = 2 * n * lambda_
    identity = np.eye(tX.shape[1])
    w = np.linalg.inv(tX.T @ tX + lambda_hat*identity) @ tX.T @ y
    loss = 1/(2*n) * np.linalg.norm(tX @ w - y)**2 + lambda_ * np.linalg.norm(w)**2
    
    return (w, loss)


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


def logistic_regression_loss(y, tx, w, lambda_=0):
    """compute the loss: negative log likelihood."""
    
    loss = np.sum(np.log(1+np.exp(tx @ w)) - y * (tx @ w)) + lambda_/2 * np.dot(w.T,w)
    return loss


def logistic_regression_gradient(y, tx, w, lambda_=0):
    """compute the gradient of loss."""
    
    grad = tx.T @ (sigmoid(tx @ w) - y) + lambda_ * w
    return grad


def logistic_regression(y, tx, initial_w, max_iter, gamma):
    '''Gradient Descent for logistic regression method
    
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
    
    w = initial_w.copy()
    
    for iter_ in range(max_iter):
        loss = logistic_regression_loss(y, tx, w)
        grad = logistic_regression_gradient(y, tx, w)        
        w -= gamma * grad

    return (w, loss)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iter, gamma):
    '''Gradient Descent for regularized logistic regression
    
    Args:
        y (np.array): true predictions
        tX (np.array): data features
        lambda_ (float): coefficient for regression
        initial_w (np.array): initial model weights
        max_iters (int): number of iterations
        gamma (float): learning rate parameter
    Returns:
        w (np.array): model weigths after the last iteration
        loss (float): loss value after the last iteration on corresponding batch
    '''
    
    w = initial_w.copy()
    
    for iter_ in range(max_iter):
        loss = logistic_regression_loss(y, tx, w, lambda_)
        grad = logistic_regression_gradient(y, tx, w, lambda_)        
        w -= gamma * grad

    return (w, loss)