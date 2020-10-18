import numpy as np


def least_squares(y, tX):
    """Finds optimal solution for linear regression using normal equations

    Args:
        y (nd.array): true predictions
        tX (nd.array): data features

    Returns:
        Tuple (nd.array, float), where first is parameters and second loss value
    """
    assert y.shape[0]==tX.shape[0], "First dimenstion of y doesn't match first dimentsion of tX"

    #Adds bias parameters
    x = tX.reshape(y.shape[0], -1)
    x = np.hstack((np.ones(x.shape[0], 1), x))

    w = np.linalg.pinv(x) @ y
    loss = np.linalg.norm(x @ w - y)**2

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

    #Adds bias parameters
    x = tX.reshape(y.shape[0], -1)
    x = np.hstack((np.ones(tX.shape[0], 1), tX))
    identity = np.eye(x.shape[0])
    identity[0, 0] = 0

    w = np.linalg.pinv(x.T @ x + lambda_*identity) @ x.T @ y
    loss = np.linalg.norm(w @ x - y)**2 + lambda_ * np.linalg.norm(w[1:, ...])**2
    
    return (w, loss)