import numpy as np

from least_squares_GD import compute_least_squares_loss, compute_least_squares_gradient


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    '''Stochastic Gradient Descent for Least Squares method using batch size = 1
    
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
    w = initial_w
    batch_numbers = np.random.randint(0, y.shape[0], size=max_iters)
    for i in batch_numbers:
        grad = compute_least_squares_gradient(y[i:i+1], tx[i:i+1], w)
        w = w - gamma*grad
    
    loss = compute_least_squares_loss(y[i:i+1], tx[i:i+1], w)
    return (w, loss)
