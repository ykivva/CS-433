import numpy as np

from least_squares_GD import compute_least_squares_loss, compute_least_squares_gradient


def least_squares_SGD(y, tX, initial_w, max_iters, lr):
    '''Stochastic Gradient Descent for Least Squares method using batch size = 1
    
    Args:
        y (np.array): true predictions
        tX (np.array): data features
        initial_w (np.array): initial model weights
        max_iters (int): number of iterations
        lr (float): learning rate parameter
    Returns:
        w (np.array): model weigths after the last iteration
        loss (float): loss value after the last iteration on corresponding batch
    '''
    assert y.shape[0]==tX.shape[0], "First dimenstion of y doesn't match first dimentsion of tX"
    
    w = initial_w
    batch_numbers = np.random.randint(0, y.shape[0], size=max_iters)
    for i in batch_numbers:
        grad = compute_least_squares_gradient(y[i:i+1], tX[i:i+1], w)
        w = w - lr*grad
    
    last_batch = batch_numbers[-1]
    loss = compute_least_squares_loss(y[last_batch:last_batch+1], tX[last_batch:last_batch+1], w)
    return (w, loss)
