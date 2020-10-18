import numpy as np


def sigmoid(x):
    res = x.copy()
    res[res > 0] = 1. / (1 + np.exp(-res[res>0]))
    res[res < 0] = np.exp(res[res < 0]) / (1 + np.exp(res[res < 0]))
    return res


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

    assert y.shape[0]==tx.shape[0], "First dimenstion of y doesn't match first dimentsion of tx"
    assert tx.shape[1]==w.shape[0], "Second dimention of x doesn't match first dimention of w" 

    h = tx @ w
    output = sigmoid(h)
    
    print(output)
    loss = -np.dot(y,  np.log(output)) - np.dot(1 - y, np.log(1 - output)) + lambda_ * np.linalg.norm(w[1:, ...])**2
    print(loss)
    
    y_mat = y.reshape(y.shape[0], -1)
    doutput = y_mat / (output + epsilon) + (1 - y_mat) / (1 - output + epsilon)
    dh = doutput * output * (1 - output)
    dw = tx.T @ dh
    grads = dw
    grads[1:, ...] += 2 * lambda_ * w[1:, ...] 

    return (loss, grads)


def logistic_regression(y, tx, initial_w, max_iter, gamma, batch_size=1):
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
    assert y.shape[0]==tx.shape[0], "First dimenstion of y doesn't match first dimentsion of tx"

    #Adds bias parameters
    x = tx.reshape(y.shape[0], -1)
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    w = initial_w
    assert x.shape[1]==w.shape[0], "Second dimention of x doesn't match first dimention of w" 

    for iter_ in range(max_iter):
        sample_num = np.random.choice(tx.shape[0], size=batch_size)
        x_sgd = x[sample_num]
        y_sgd = y[sample_num]
        loss, grads = forward_backward(y_sgd, x_sgd, w)
        w -= gamma * grads

    return (w, loss)


def logistic_regression_reg(y, tx, lambda_, initial_w, max_iter, gamma, batch_size=1, verbose=1):
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
    #Adds bias parameters
    x = tx.reshape(y.shape[0], -1)
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    assert x.shape[1]==initial_w.shape[0], "Second dimention of x doesn't match first dimention of w" 

    w = initial_w

    for iter_ in range(max_iter):
        
        sample_num = np.random.choice(tx.shape[0], size=batch_size)
        x_sgd = x[sample_num]
        y_sgd = y[sample_num]
        loss, grads = forward_backward(y_sgd, x_sgd, w, lambda_)
        w -= gamma * grads
        if verbose==1:
                bar  = (iter_*20//max_iter)*"#" + " " * (20 - (iter_*20//max_iter))
                print(f'\r>Iter #{iter_}:\t[{bar}]; Loss: {loss}', end='')

    return (w, loss)


def logistic_pred(x, w):
    y_prob = sigmoid(x @ w)
    y_pred = y_prob > 0.5
    return y_pred