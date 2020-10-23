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