import numpy as np


def sigmoid(x):
    res = x.copy()
    res[res > 0] = 1. / (1 + np.exp(-res[res>0]))
    res[res < 0] = np.exp(res[res < 0]) / (1 + np.exp(res[res < 0]))
    return res


def forward_backward(y, tX, w, lambda_=0):
    '''Compute gradient of Least Squares loss function

    Args:
        y (np.array): true predictions
        tX (np.array): data features
        w (np.array): model weights

    Returns:
        grads (np.array): gradient with respect to model weights
    '''
    epsilon = 10**(-100)

    assert y.shape[0]==tX.shape[0], "First dimenstion of y doesn't match first dimentsion of tX"
    assert tX.shape[1]==w.shape[1], "Second dimention of x doesn't match second dimention of w" 
    
    h = tX @ w.T
    output = sigmoid(h)
    
    assert not np.any(np.isnan(tX)), "Nan found in features"
    assert not np.any(np.isnan(w)), "Nan found in w"
    assert not np.any(np.isnan(h)), "Nan found in hidden layer"
    assert not np.any(np.isnan(output)), "Nan found among activations"
    
    loss = -np.dot(y,  np.log(output+epsilon).squeeze()) - np.dot(1 - y, np.log(1 - output+epsilon).squeeze())
    loss /= y.shape[0]
    
    y_mat = y.reshape(output.shape)
    dh = output - y_mat
    dw = dh.T @ tX
    grads = dw
    grads[:, 1:] += 2 * lambda_ * w[:, 1:] 

    return (loss, grads)


def logistic_regression(y, tX, initial_w, max_iter, lr, batch_size=1):
    '''Stochastic Gradient Descent for logistic regression method
    
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

    #Adds bias parameters
    x = tX.reshape(y.shape[0], -1)
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    w = initial_w.copy()
    assert x.shape[1]==w.shape[1], "Second dimention of x doesn't match second dimention of w" 

    for iter_ in range(max_iter):
        sample_num = np.random.choice(tX.shape[0], size=batch_size)
        x_sgd = x[sample_num]
        y_sgd = y[sample_num]
        loss, grads = forward_backward(y_sgd, x_sgd, w)
        w -= lr * grads

    return (w, loss)


def logistic_regression_reg(y, tX, lambda_, initial_w, max_iter, lr, batch_size=1, verbose=1):
    '''Stochastic Gradient Descent for logistic regression method using batch size = 1
    
    Args:
        y (np.array): true predictions
        tX (np.array): data features
        labbda_ (float): coefficient for regression
        initial_w (np.array): initial model weights
        max_iters (int): number of iterations
        lr (float): learning rate parameter
    Returns:
        w (np.array): model weigths after the last iteration
        loss (float): loss value after the last iteration on corresponding batch
    '''
    #Adds bias parameters
    x = tX.reshape(y.shape[0], -1)
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    assert x.shape[1]==initial_w.shape[1], "Second dimention of x doesn't match second dimention of w" 

    w = initial_w.copy()

    for iter_ in range(max_iter):
        
        sample_num = np.random.choice(tX.shape[0], size=batch_size, replace=False)
        x_sgd = x[sample_num]
        y_sgd = y[sample_num]
        loss, grads = forward_backward(y_sgd, x_sgd, w, lambda_)
        w = w - lr * grads
        if verbose==1:
            bar  = (iter_*20//max_iter)*"#" + " " * (20 - (iter_*20//max_iter))
            print(f'\r>Iter #{iter_}:\t[{bar}]; Loss: {loss}, {lr}', end='')

    return (w, loss)


def logistic_pred(tX, w):
    x = tX.reshape(tX.shape[0], -1)
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    y_prob = sigmoid(x @ w.T)
    y_pred = y_prob > 0.5
    return y_pred